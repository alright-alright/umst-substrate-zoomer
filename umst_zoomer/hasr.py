from .loop import Edge

def hasr_update(edges, is_bound, lr=0.05, decay=0.002):
    # Reinforce edges whose both endpoints are bound, else decay slightly
    for e in edges:
        if is_bound[e.i] and is_bound[e.j]:
            e.w += lr * (1.0 - e.w)
        else:
            e.w *= (1.0 - decay)
    return edges
