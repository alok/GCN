module GCN

using LightGraphs, MetaGraphs, SimpleWeightedGraphs

A = [
    [0., 1., 0., 0., 0.],
    [1., 0., 1., 0., 0.],
    [0., 1., 0., 1., 1.],
    [0., 0., 1., 0., 0.],
    [0., 0., 1., 0., 0.],
]

feats = reshape(1:5, :, 1)

function karate_club_graph()
    # Create the set of all members, and the members of each club.
    include("karate_adjacency_matrix.jl")
    G = Graph(adjacency_mat)

end

end
