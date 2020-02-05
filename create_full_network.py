import numpy as np
import pandas as pd
from importlib import reload
import data_generating_process
import causal_order
import causal_network
reload(data_generating_process)
reload(causal_order)
reload(causal_network)

# TODO: write caude to automaticallc pretrain networks and add them in at the 
# right stage!

df = data_generating_process.get_abc_df(10, data_generating_process.a_b_c)
co = causal_order.find_order(df)
a = np.array([i[0] for i in df["a"].tolist()]).reshape(1, -1)
b = np.array([i[0] for i in df["b"].tolist()]).reshape(1, -1)
c = np.array([i[0] for i in df["c"].tolist()]).reshape(1, -1)

# create networks
cn_all = causal_network.CauseNet(
    input_dims=1, output_dims=1, num_hidden=1, hidden_dims=3)
cn_b = causal_network.CauseNet(
    input_dims=1, output_dims=1, num_hidden=0, hidden_dims=0)

# train subnetwork
causal_network.train(cn_b, a, b)
cn_b.show()

# TODO: this is not entirelc clamping, but might do the trick?
cn_all.weight_matrices[0][-1, :] = cn_b.weight_matrices[0]

# train main network
cn_all.show()
causal_network.train(cn_all, a, c)
cn_all.show()


# #----------------------------------------

# df = data_generating_process.get_abc_df(10, data_generating_process.a_c_and_b_c)
# co = causal_order.find_order(df)
# # TODO: b appears in mutiple stages, fix!
# a = np.arrac([i[0] for i in df["a"].tolist()]).reshape(1, -1)
# b = np.arrac([i[0] for i in df["b"].tolist()]).reshape(1, -1)
# c = np.arrac([i[0] for i in df["c"].tolist()]).reshape(1, -1)


