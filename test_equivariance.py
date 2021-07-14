import gvp
import gvp.models
import gvp.data
import torch
from torch import nn
from scipy.spatial.transform import Rotation
import unittest

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node_dim = (100, 16)
edge_dim = (32, 1)
n_nodes = 300
n_edges = 10000

nodes = gvp.randn(n_nodes, node_dim, device=device)
edges = gvp.randn(n_edges, edge_dim, device=device)
edge_index = torch.randint(0, n_nodes, (2, n_edges), device=device)
batch_idx = torch.randint(0, 5, (n_nodes,), device=device)
seq = torch.randint(0, 20, (n_nodes,), device=device)

class EquivarianceTest(unittest.TestCase):

    def test_gvp(self):
        model = gvp.GVP(node_dim, node_dim).to(device).eval()
        model_fn = lambda h_V, h_E: model(h_V)
        test_equivariance(model_fn, nodes, edges)
        
    def test_gvp_vector_gate(self):
        model = gvp.GVP(node_dim, node_dim, vector_gate=True).to(device).eval()
        model_fn = lambda h_V, h_E: model(h_V)
        test_equivariance(model_fn, nodes, edges)

    def test_gvp_sequence(self):
        model = nn.Sequential(
            gvp.GVP(node_dim, node_dim),
            gvp.Dropout(0.1),
            gvp.LayerNorm(node_dim)
        ).to(device).eval()
        model_fn = lambda h_V, h_E: model(h_V)
        test_equivariance(model_fn, nodes, edges)
        
    def test_gvp_sequence_vector_gate(self):
        model = nn.Sequential(
            gvp.GVP(node_dim, node_dim, vector_gate=True),
            gvp.Dropout(0.1),
            gvp.LayerNorm(node_dim)
        ).to(device).eval()
        model_fn = lambda h_V, h_E: model(h_V)
        test_equivariance(model_fn, nodes, edges)
        
    def test_gvp_conv(self):
        model = gvp.GVPConv(node_dim, node_dim, edge_dim).to(device).eval()
        model_fn = lambda h_V, h_E: model(h_V, edge_index, h_E)
        test_equivariance(model_fn, nodes, edges)
        
    def test_gvp_conv_vector_gate(self):
        model = gvp.GVPConv(node_dim, node_dim, edge_dim, vector_gate=True).to(device).eval()
        model_fn = lambda h_V, h_E: model(h_V, edge_index, h_E)
        test_equivariance(model_fn, nodes, edges)
        
    def test_gvp_conv_layer(self):
        model = gvp.GVPConvLayer(node_dim, edge_dim).to(device).eval()
        model_fn = lambda h_V, h_E: model(h_V, edge_index, h_E,
                                          autoregressive_x=h_V)
        test_equivariance(model_fn, nodes, edges)
        
    def test_gvp_conv_layer_vector_gate(self):
        model = gvp.GVPConvLayer(node_dim, edge_dim, vector_gate=True).to(device).eval()
        model_fn = lambda h_V, h_E: model(h_V, edge_index, h_E,
                                          autoregressive_x=h_V)
        test_equivariance(model_fn, nodes, edges)
        
    def test_mqa_model(self):
        model = gvp.models.MQAModel(node_dim, node_dim, 
                                    edge_dim, edge_dim).to(device).eval()
        model_fn = lambda h_V, h_E: (model(h_V, edge_index, h_E, batch=batch_idx), \
                                     torch.zeros_like(nodes[1]))
        test_equivariance(model_fn, nodes, edges)
    
    def test_cpd_model(self):
        model = gvp.models.CPDModel(node_dim, node_dim, 
                                    edge_dim, edge_dim).to(device).eval()
        model_fn = lambda h_V, h_E: (model(h_V, edge_index, h_E, seq=seq), \
                                     torch.zeros_like(nodes[1]))
        test_equivariance(model_fn, nodes, edges)
    
    
def test_equivariance(model, nodes, edges):
    
    random = torch.as_tensor(Rotation.random().as_matrix(), 
                             dtype=torch.float32, device=device)
    
    with torch.no_grad():
    
        out_s, out_v = model(nodes, edges)
        n_v_rot, e_v_rot = nodes[1] @ random, edges[1] @ random
        out_v_rot = out_v @ random
        out_s_prime, out_v_prime = model((nodes[0], n_v_rot), (edges[0], e_v_rot))
        
        assert torch.allclose(out_s, out_s_prime, atol=1e-5, rtol=1e-4)
        assert torch.allclose(out_v_rot, out_v_prime, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()