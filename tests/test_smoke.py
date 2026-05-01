from seedgpt.config import SEEDGPTConfig
from seedgpt.data import generate_synthetic_graphs, collate_graphs
from seedgpt.models import GINEncoder
from seedgpt.agent import SEEDGPTAgent


def test_forward_smoke():
    cfg = SEEDGPTConfig(num_graphs=8, min_nodes=6, max_nodes=10, hidden_dim=16, num_gnn_layers=2, policy_hidden_dim=16)
    graphs = generate_synthetic_graphs(4, cfg.num_node_features, cfg.min_nodes, cfg.max_nodes, seed=1)
    batch = collate_graphs(graphs)
    encoder = GINEncoder(cfg.num_node_features, cfg.hidden_dim, cfg.num_gnn_layers)
    agent = SEEDGPTAgent(cfg)
    prompt = agent.initial_prompt(batch.x, batch.mask)
    node_repr, graph_repr = encoder(batch.x + prompt, batch.adj, batch.mask)
    state = agent.state_from_node_repr(node_repr, batch.mask)
    node, edit, logp = agent.principal.act(state, node_repr, batch.mask)
    assert prompt.shape == batch.x.shape
    assert node.shape[0] == batch.x.shape[0]
    assert edit.shape[-1] == cfg.num_node_features
    assert logp.shape[0] == batch.x.shape[0]
