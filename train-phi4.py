import tensorflow as tf
from phi4models import G, GN
import argparse
import json
import time
import sys
tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx('float32')

default_sample_size = 2**13

parser = argparse.ArgumentParser()
parser.add_argument("ens_id", metavar="ENS_ID", type=str, help="Ensemble ID")
parser.add_argument("run_id", metavar="RUN_ID", type=str, help="Run ID")
parser.add_argument("-n", action="store_true", default=False, help="Start a new training")
parser.add_argument("-c", metavar="I", action="store", default=None, help="Continue from iteration ")
parser.add_argument("-l", metavar="L", action="store", type=float, default=1e-3, help="Learning rate ")
parser.add_argument("-o", metavar="D", default="./", action="store", help="Store/read checkpoints in/from directory D")
parser.add_argument("-s", metavar="S", type=int, default=default_sample_size, action="store", help="Sample size to train on (default: {})".format(default_sample_size))

args = parser.parse_args(sys.argv[1:])
ens_id = args.ens_id
run_id = args.run_id
new_train = args.n
lr = args.l
continue_iter = None if args.c == None else int(args.c)
top_dir = args.o
sample_size = args.s

if ((continue_iter is None) and (new_train == False)):
    parser.print_usage()
    print("Need to specify either -c or -n")
    parser.exit()

def train(z):
    with tf.GradientTape() as tape:
        f,j,s = g(z)
        lo = tf.reduce_mean(tf.add(tf.add(j, s), g.log_r(z)))
    grads = tape.gradient(lo, g.trainable_variables)
    optim.apply_gradients(zip(grads, g.trainable_variables))
    return lo

def r_acc(s, logpf=None):
    s0 = s[0]
    ri = tf.Variable([0]*s.shape[0])
    j = 0
    for i in range(s.shape[0]):
        if logpf is not None:
            ds = -(s[i]-s0)+(logpf[j]-logpf[i])
        else:
            ds = -(s[i]-s0)
        if ds.numpy() > 0:
            ri[i].assign(1)
            s0 = s[i]
            j = i
        elif tf.random.uniform(()) < tf.exp(ds):
            ri[i].assign(1)
            s0 = s[i]
            j = i
    return tf.reduce_mean(tf.cast(ri, tf.float32))
    
ensembles = json.load(open("ensembles.json", "r"))["ensembles"]
nn_params = json.load(open("nn-params.json", "r"))["nn-params"]

ens = ensembles[ens_id]
L,m2,la = ens["L"], ens["m2"], ens["la"]
logfile = "{}-{}-log.txt".format(ens_id, run_id)
if new_train:
    i = 0
    N,nodes = nn_params[ens_id]["N"], nn_params[ens_id]["nodes"]
    g = GN(N, nodes, L, m2, la)
    # truncate logfile
    with open(logfile, "w") as fp:
        pass
else:
    i = continue_iter+1
    N = nn_params[ens_id]["N"]
    g = GN.from_file("{}/{}/{}/iter{:06.0f}".format(top_dir, ens_id, run_id, continue_iter), N, L, m2, la)
    line = "# Restarting iter = {:6.0f}".format(continue_iter)
    print(line)
    with open(logfile, "a") as fp:
        fp.writelines(line + "\n")
optim = tf.keras.optimizers.Adam(learning_rate=lr)

lines = list()
while True:
    z = tf.random.normal([sample_size,L*L])
    lo = train(z)
    if i % 250 == 0:
        z0 = tf.random.normal([sample_size,L*L])
        s = g.S(g.forward(z0))
        logpf = g.logdetjacfwd(z0) + g.log_r(z0)
        acc = r_acc(s, logpf)
        line = " {} iter = {:6.0f}, loss = {:18.4f}, acc = {:6.3f}".format(time.strftime("[%Y-%m-%d %H:%M:%S]"), i, lo, acc)
    else:
        line = " {} iter = {:6.0f}, loss = {:18.4f}".format(time.strftime("[%Y-%m-%d %H:%M:%S]"), i, lo)
    print(line)
    lines.append(line+"\n")
    if i % 10 == 0:
        with open(logfile, "a") as fp:
            fp.writelines(lines)
        lines = list()
    # Checkpoint
    if (i % 500 == 0) and (i != 0):
        dirname = "{}/{}/iter{:06.0f}".format(ens_id, run_id, i)
        print("# -- Checkpointing --")
        g.save(top_dir + "/" + dirname)
        print("# wrote to: {}".format(dirname))
    i += 1
