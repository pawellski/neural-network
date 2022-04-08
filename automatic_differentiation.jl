# structs definition
abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name :: String
    Variable(output; name="?") = new(output, nothing, name)
end

mutable struct VectorOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    VectorOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

# nodes printing
import Base: show, summary
show(io::IO, x::VectorOperator{F}) where {F} = begin
    print(io, "VectorOp.", x.name, "(", F, ")");
end
show(io::IO, x::Constant) = print(io, "Constant ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "Variable ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
end

# graphs building
function visit(node::GraphNode, visited, order)
    if node ∉ visited
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::Operator, visited, order)
    if node ∉ visited
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(root::GraphNode)
    visited = Set()
    order = Vector()
    visit(root, visited, order)
    return order
end

# forward pass
reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

# backward pass
update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) =
    if isnothing(node.gradient)
        node.gradient = gradient
    else
        node.gradient .+= gradient
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end

# operations
Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = VectorOperator(+, x, y)
forward(::VectorOperator{typeof(+)}, x, y) = return x .+ y
backward(::VectorOperator{typeof(+)}, x, y, g) = tuple(g, g)

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = VectorOperator(-, x, y)
forward(::VectorOperator{typeof(-)}, x, y) = return x .- y
backward(::VectorOperator{typeof(-)}, x, y, g) = return tuple(g, -g)

import Base: *
import LinearAlgebra: mul!
*(A::GraphNode, x::GraphNode) = VectorOperator(mul!, A, x)
forward(::VectorOperator{typeof(mul!)}, A, x) = return A * x
backward(::VectorOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = VectorOperator(*, x, y)
forward(::VectorOperator{typeof(*)}, x, y) = return x .* y
backward(node::VectorOperator{typeof(*)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(y .* 𝟏)
    Jy = diagm(x .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = VectorOperator(/, x, y)
forward(::VectorOperator{typeof(/)}, x, y) = return x ./ y
backward(node::VectorOperator{typeof(/)}, x, y::Real, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(𝟏 ./ y)
    Jy = (-x ./ y .^2)
    tuple(Jx' * g, Jy' * g)
end

import Base: ^
Base.Broadcast.broadcasted(^, x::GraphNode, n::GraphNode) = VectorOperator(^, x, n)
forward(::VectorOperator{typeof(^)}, x, n) = return x .^ n
backward(::VectorOperator{typeof(^)}, x, n, g) = let
    𝟏 = ones(length(x))
    Jx = diagm(n .* x .^ (n-1) .* 𝟏)
    Jn = log.(abs.(x)) .* x .^ n
    tuple(Jx' * g, Jn' * g)
end

import Base: sum
sum(x::GraphNode) = VectorOperator(sum, x)
forward(::VectorOperator{typeof(sum)}, x) = return sum(x)
backward(::VectorOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
    tuple(J' * g)
end

import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = VectorOperator(max, x, y)
forward(::VectorOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::VectorOperator{typeof(max)}, x, y, g) = let
    Jx = diagm(isless.(y, x))
    Jy = diagm(isless.(x, y))
    tuple(Jx' * g, Jy' * g)
end

linear(x) = return x
Base.Broadcast.broadcasted(linear, x::GraphNode) = VectorOperator(linear, x)
forward(::VectorOperator{typeof(linear)}, x) = return x
backward(::VectorOperator{typeof(linear)}, x, g) = let
    𝟏 = ones(length(x), length(x))
    tuple(𝟏' * g)
end

ReLU(x) = return x
Base.Broadcast.broadcasted(ReLU, x::GraphNode) = VectorOperator(ReLU, x)
forward(::VectorOperator{typeof(ReLU)}, x) = return max.(zeros(length(x)), x)
backward(::VectorOperator{typeof(ReLU)}, x, g) = let
    𝟏 = ones(length(x))
    x[x .<= zero(x)] .= 0.0
    x[x .!= zero(x)] .= 1.0
    Jx = diagm(x .* 𝟏)
    tuple(Jx' * g)
end

σ(x) = return x
Base.Broadcast.broadcasted(σ, x::GraphNode) = VectorOperator(σ, x)
forward(::VectorOperator{typeof(σ)}, x) = return 1 ./ (1 .+ ℯ .^ (.- x))
backward(::VectorOperator{typeof(σ)}, x, g) = let
    𝟏 = ones(length(x))
    Jx = diagm(((ℯ .^ (.-x))./((1 .+ ℯ .^ (.-x)) .^ 2)) .* 𝟏)
    tuple(Jx' * g)
end

softmax(x) = return x
Base.Broadcast.broadcasted(softmax, x::GraphNode) = VectorOperator(softmax, x)
forward(::VectorOperator{typeof(softmax)}, x) =  (ℯ .^ (x))./(sum(ℯ .^x))
backward(::VectorOperator{typeof(softmax)}, x, g) =  let
    v = (ℯ .^ (x))./(sum(ℯ .^x))
    len = length(v)
    Jx = ones(len, len)
    for i in 1:len
        for j in 1:len
            if i == j setindex!(Jx, v[i] * (1-v[i]), i, i)
                 else setindex!(Jx, -v[i] * v[j], i, j) end
        end
    end
    tuple(Jx' * g)
end
