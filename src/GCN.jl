# module GCN
using LightGraphs, MetaGraphs, SimpleWeightedGraphs, LinearAlgebra, Flux, GeometricFlux

A = [
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
]

feats = reshape(1:5, :, 1)

function karate_club_graph()
    # Create the set of all members, and the members of each club.
    include("karate_adjacency_matrix.jl")
    G = Graph(adjacency_mat)
end

g = MetaGraph(karate_club_graph())
communities, _ = label_propagation(g)
for (v, community) in enumerate(communities)
    set_prop!(g, v, :community, community)
end

n_classes = length(unique(communities))
labels = onehotbatch(communities, 1:n_classes)

model = Chain(
    GCNConv(g, 1024 => 512, relu),
    Dropout(0.5),
    GCNConv(g, 512 => 128),
    Dense(128, 10),
    softmax,
)

#TODO impl on abstractgraph
# communities=greedy_modularity_communities(g)

## Loss
loss(x, y) = logitcrossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

## Training
ps = Flux.params(model)
train_X = 1:3
train_Y = [get_prop(g, v, :community) for v in train_X]
train_data = [(train_X, train_y)]

opt = ADAM(0.01)
evalcb() = @show(accuracy(train_X, train_y))

train!(loss, ps, train_data, opt, cb = throttle(evalcb, 10))

# end
