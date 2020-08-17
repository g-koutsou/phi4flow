import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import math
import os

class G(Model):
    def __init__(self, evenodd, nnodes, L):
        super(G, self).__init__()
        assert evenodd in [0,1]
        assert (L*L) % 2 == 0
        self.s = tf.keras.Sequential()
        self.t = tf.keras.Sequential()
        nlayers = len(nnodes)
        self.N = nlayers
        self.L = L
        y0,x0 = tf.meshgrid(list(range(L)),list(range(L)))
        self.idx = [tf.cast(tf.reshape(tf.where(tf.equal(tf.reshape(y0 + x0, [L*L]) % 2, (evenodd+0)%2)), [(L*L)//2]), tf.int32),
                    tf.cast(tf.reshape(tf.where(tf.equal(tf.reshape(y0 + x0, [L*L]) % 2, (evenodd+1)%2)), [(L*L)//2]), tf.int32),]
        self.s.add(tf.keras.layers.InputLayer(input_shape=((L*L)//2,)))
        self.t.add(tf.keras.layers.InputLayer(input_shape=((L*L)//2,)))
        for i in range(1,nlayers+1):
            self.s.add(tf.keras.layers.Dense(nnodes[i-1], activation=tf.nn.leaky_relu, use_bias=False))
            self.t.add(tf.keras.layers.Dense(nnodes[i-1], activation=tf.nn.leaky_relu, use_bias=False))
        self.s.add(tf.keras.layers.Dense((L*L)//2))#, activation=tf.nn.leaky_relu, use_bias=False))
        self.t.add(tf.keras.layers.Dense((L*L)//2))#, activation=tf.nn.leaky_relu, use_bias=False))

    def forward(self, z):
        z_a = tf.gather(z, self.idx[0], axis=1)
        z_b = tf.gather(z, self.idx[1], axis=1)
        f_a = z_a
        s_a = tf.math.exp(-self.s(z_a))
        t_a = self.t(z_a)
        f_b = tf.math.multiply(tf.math.subtract(z_b, t_a), s_a)
        i_a = tf.transpose(tf.meshgrid(list(range(z.shape[0])), self.idx[0], indexing="ij"), (1,2,0))
        i_b = tf.transpose(tf.meshgrid(list(range(z.shape[0])), self.idx[1], indexing="ij"), (1,2,0))
        out = tf.add(tf.scatter_nd(i_a, f_a, z.shape), tf.scatter_nd(i_b, f_b, z.shape))
        return out

    def inverse(self, f):
        f_a = tf.gather(f, self.idx[0], axis=1)
        f_b = tf.gather(f, self.idx[1], axis=1)
        z_a = f_a
        s_a = tf.math.exp(self.s(f_a))
        t_a = self.t(f_a)
        z_b = tf.math.add(tf.math.multiply(f_b, s_a), t_a)
        i_a = tf.transpose(tf.meshgrid(list(range(f.shape[0])), self.idx[0], indexing="ij"), (1,2,0))
        i_b = tf.transpose(tf.meshgrid(list(range(f.shape[0])), self.idx[1], indexing="ij"), (1,2,0))
        out = tf.add(tf.scatter_nd(i_a, z_a, f.shape), tf.scatter_nd(i_b, z_b, f.shape))
        return out

    def logdetjacinv(self, f):
        f_a = tf.gather(f, self.idx[0], axis=1)
        exp = self.s(f_a)
        return tf.math.reduce_sum(exp, axis=1)

    def logdetjacfwd(self, z):
        z_a = tf.gather(z, self.idx[0], axis=1)
        exp = self.s(z_a)
        return tf.math.reduce_sum(exp, axis=1)
    
    def call(self, z):
        f = self.forward(z)
        d = self.logdetjacfwd(z)
        return f,d

class GN(Model):
    def __init__(self, N, nodes, L, m2, la):
        super(GN, self).__init__()
        self.N = N
        self.m2 = tf.cast(m2, tf.float32)
        self.la = tf.cast(la, tf.float32)
        self.L = L
        self.g = list([None]*N)
        for i in range(N):
            self.g[i] = G(i % 2, nodes[i], L)
    @classmethod
    def from_file(cls, dirname, N, L, m2, la):
        s,t = list(),list()
        for i in range(N):
            s.append(tf.keras.models.load_model("{}/s{}".format(dirname, i), custom_objects={'leaky_relu': tf.nn.leaky_relu}))
            t.append(tf.keras.models.load_model("{}/t{}".format(dirname, i), custom_objects={'leaky_relu': tf.nn.leaky_relu}))
        # For each affine layer, s and t should have the same architecture
        s_nodes = [[l['config']['units'] for l in x.get_config()['layers']] for x in s]
        t_nodes = [[l['config']['units'] for l in x.get_config()['layers']] for x in t]
        assert s_nodes == t_nodes
        # Check if output has the right size
        for sn in s_nodes:
            assert sn[-1] == (L*L)//2
        g = cls(N, [sn[:-1] for sn in s_nodes], L, m2, la)
        for i in range(N):
            g.g[i].s = s[i]
            g.g[i].t = t[i]
        return g
    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        for i in range(self.N):
            self.g[i].s.save("{}/s{}".format(dirname, i))
            self.g[i].t.save("{}/t{}".format(dirname, i))
    def forward(self, z):
        f = tf.identity(z)
        for i in range(self.N):
            f = self.g[i].forward(f)
        return f
    def forward_and_logdetjacfwd(self, z):
        f = tf.identity(z)
        de = self.g[0].logdetjacfwd(z)
        f = self.g[0].forward(z)
        for i in range(1,self.N):
            de = tf.add(self.g[i].logdetjacfwd(f), de)
            f = self.g[i].forward(f)
        return f,de
    def inverse(self, f):
        z = tf.identity(f)
        for i in reversed(range(self.N)):
            z = self.g[i].inverse(z)
        return z
    def logdetjacinv(self, f):
        de = self.g[-1].logdetjacinv(f)
        z = tf.identity(f)
        for i in reversed(range(self.N-1)):
            z = self.g[i+1].inverse(z)
            de = tf.add(self.g[i].logdetjacinv(z), de)
        return de
    def logdetjacfwd(self, z):
        de = self.g[0].logdetjacfwd(z)
        f = tf.identity(z)
        for i in range(1,self.N):
            z = self.g[i-1].forward(z)
            de = tf.add(self.g[i].logdetjacfwd(z), de)
        return de
    def S(self, f):
        L, m2,la = self.L, self.m2, self.la
        f00 = tf.reshape(tf.identity(f), (-1, L, L))
        f0p = tf.roll(f00, axis=2, shift=+1)
        f0m = tf.roll(f00, axis=2, shift=-1)
        fp0 = tf.roll(f00, axis=1, shift=+1)
        fm0 = tf.roll(f00, axis=1, shift=-1)
        D = tf.add(tf.add(tf.add(f0p, f0m), fp0), fm0)
        D = tf.subtract(tf.multiply((4.0+m2), f00), D)
        D = tf.multiply(f00, D)
        s = tf.add(D,  tf.multiply(la, tf.pow(f00, 4)))
        return tf.reduce_sum(tf.reshape(s, (-1, L*L)), axis=1)
    def log_r(self, z):
        L = self.L
        return -tf.reduce_sum(z**2, axis=1)*0.5 - ((L*L)//2)*tf.math.log(2*math.pi)
    def call(self, z):
        f,d = self.forward_and_logdetjacfwd(z)
        s = self.S(f)
        return f,d,s
