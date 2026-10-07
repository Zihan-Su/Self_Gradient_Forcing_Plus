"""Angular-distance t-SNE of context and generation coordinate sketches."""
from sklearn.manifold import TSNE
import json
import numpy as np
import torch
from plot_utils import FAMILIES, inputs, load, save
import paper_style


def main():
    p = inputs(__doc__)
    p.add_argument('--perplexity', type=float, default=20)
    p.add_argument('--seed', type=int, default=23092026)
    a = p.parse_args()
    rows, groups, metrics = load(a.data)
    count = len(rows)
    if not 0 < a.perplexity < 2*count:
        raise ValueError('Perplexity must be positive and smaller than point count')
    metrics.update(method='t-SNE on sketch-estimated normalized angular distance',
                   perplexity=a.perplexity, seed=a.seed, families={})
    embeddings = {}
    for family in FAMILIES:
        gram = np.zeros((2*count, 2*count), dtype=np.float64)
        metadata = json.loads((a.data/f"prompt_{rows[0]['prompt_index']:03d}_protocol.json").read_text())['modules']
        names = sorted(groups[family], key=lambda n: (metadata[n]['layer'], metadata[n]['family'], metadata[n]['projection']))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for name in names:
            stats = [r['module_stats'][name] for r in rows]
            v = torch.stack([s['sketch_context'] for s in stats] +
                            [s['sketch_generation'] for s in stats]).to(device)
            gram += (v @ v.T).double().cpu().numpy()
        norm = np.sqrt(np.diag(gram))
        if (norm <= 0).any():
            raise ValueError('Zero grouped sketch norm')
        distance = np.arccos(np.clip(gram/np.outer(norm, norm), -1, 1))/np.pi
        distance = (distance + distance.T)/2
        np.fill_diagonal(distance, 0)
        model = TSNE(n_components=2, perplexity=a.perplexity, metric='precomputed',
                     init='random', learning_rate='auto', method='exact',
                     max_iter=2000, random_state=a.seed)
        # sklearn mutates precomputed distances; pass a copy.
        xy = model.fit_transform(distance.copy())
        if xy[:count, 0].mean() > xy[count:, 0].mean():
            xy[:, 0] *= -1
        embeddings[family] = xy
        metrics['families'][family] = {'kl_divergence': float(model.kl_divergence_),
                                     'points_per_role': count}
    fig = paper_style.distributions(embeddings, count)
    save(fig, a.out, 'a_gradient_distributions', metrics)
    np.savez_compressed(a.out/'a_gradient_distributions_embeddings.npz', **embeddings)


if __name__ == '__main__':
    main()
