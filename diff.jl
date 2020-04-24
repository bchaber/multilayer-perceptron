abstract type Node end
abstract type Operator end
abstract type LeafNode <: Node end

mutable struct Variable{T} <: LeafNode
    value::T
    grad::T
    name::String
end
Variable(value) = Variable(value, zero(value), "")

import Base: zero, one
zero(::Variable{T}) where T = Variable(zero(T), zero(T))
one(::Variable{T}) where T = Variable(one(T), zero(T))

struct Method{OT} <: Operator
    f::OT
end

struct Broadcasted{OT} <: Operator
    f::OT
end
(op::Broadcasted)(args...; kwargs...) = op.f.(args...; kwargs...)

struct ComputableNode{OT <: Operator, AT <: Tuple, KT <: NamedTuple} <: Node
    op::OT
    args::AT
    kwargs::KT
end
ComputableNode(op::Function, args, kwargs) = ComputableNode(Method(op), args, kwargs)
ComputableNode(op, args)                   = ComputableNode(op, args, NamedTuple())

mutable struct CachedNode{NT <: Node, OUT} <: Node
    node::NT
    out::OUT
end

Tensor = Variable{T} where {T <:AbstractArray}
TensorCachedNode = CachedNode{NT, OUT} where {NT, OUT <:AbstractArray}

function register(op, args...; kwargs...)
    node = ComputableNode(op, args, kwargs.data)
    out  = forward(node)
    CachedNode(node, out)
end

arg(x::ComputableNode, i::Int) = x.args[i]
args(x::ComputableNode) = x.args
kwargs(x::ComputableNode) = x.kwargs
operator(x::ComputableNode) = x.f

arg(x::CachedNode, i::Int) = x.node.args[i]
args(x::CachedNode) = x.node.args
kwargs(x::CachedNode) = x.node.kwargs
operator(x::CachedNode) = x.node.f

import Base: size
size(x::Node) = size(value(x))

import Base: show, summary
show(io::IO, x::Method)         = print(io, "fn ",  x.f);
show(io::IO, x::Operator)       = print(io, "op ",  x.f);
show(io::IO, x::Broadcasted)    = print(io, "bc ",  x.f);

show(io::IO, x::Variable) = begin
    print(io, "var "); summary(io, x.value)
    print(io, " ∇ ");  summary(io, x.grad)
end

summary(io::IO, x::CachedNode) = print(io, "cached ", x.node.op); 
show(io::IO, x::Variable, level) = begin
    indent = repeat('\t', level)
    print(io, "\n", indent);
    print(io, "var ", x.name, " "); summary(io, x.value)
    print(io, " ∇ ");  summary(io, x.grad)
end
summary(io::IO, x::Broadcast.Broadcasted) = print(io, "broadcasted ", x.f)
show(io::IO, x, level) = begin print(io, "\n", repeat('\t', level)); show(io, x) end
show(io::IO, x::CachedNode, level) = begin
    indent = repeat('\t', level)
    print(io, "\n", indent); summary(io, x)
    for arg in x.node.args
        show(io, arg, level+1)
    end
    print(io, "\n=>", indent); summary(io, x.out)
end
show(io::IO, x::CachedNode; level=0) = show(io, x, level)

summary(io::IO, x::ComputableNode) = print(io, x.op)
show(io::IO, x::ComputableNode) = begin
    print(io, "[", x.op, "]("); summary(io, x.args);
    print(io, ")");
end

forward(cached::CachedNode) = cached.out = forward(cached.node)
forward(node::ComputableNode) = forward(node.op, map(forward, node.args)...; map(forward, node.kwargs)...)
forward(op::Method, args...; kwargs...) = op.f(args...; kwargs...)
forward(leaf::LeafNode) = value(leaf)
forward(x) = x
forward(x::NT) where {NT <: Node} = error("forward method is not implemented for node type: $NT")

value(x::CachedNode) = value(x.out)
value(x::Variable) = x.value
value(x::Tensor) = x.value
value(x) = x
value(x::NT) where {NT <: Node} = error("Expected value in this node $x of type $T
 check if you defined a non-cached node
 or overload value function for your node.")
grad(x::Variable) = x.grad
grad(x::Tensor) = x.grad
function backward(x::Variable, grad)
    x.grad += grad
    nothing
end

function backward(x::Tensor, grad)
    x.grad .+= grad
    nothing
end

function backward(cached::CachedNode, ::Any, grad)
    grad_inputs = gradient(cached, grad)
    for (each, each_grad) in zip(args(cached), grad_inputs)
        backward(each, each_grad)
    end
    nothing
end

backward(cached::CachedNode) = backward(cached, 1.0)
backward(cached::CachedNode, grad) = backward(cached, cached.node.op, grad)
backward(cached::CachedNode, op::Method, grad) = backward(cached, op.f, grad)

gradient(x::CachedNode, grad) = gradient(x.node.op, grad, x.out, map(value, x.node.args)...; map(value, x.node.kwargs)...)
gradient(x::Operator,   grad, out, args...; kwargs...) = gradient(x.f, grad, out, args...; kwargs...)
gradient(op, grad, out, args...; kwargs...) = error("gradient of operator $op is not defined\n
 Possible Fix:\n
 define one of the following:\n
 1. gradient(::typeof($op), grad, out, args...; kwargs...)\n
 2. gradient(op::Method{typeof($op)}, grad, out, args...; kwargs...)\n")

import Base: +, -, *, /
+(x::Node) = register(+, x)
-(x::Node) = register(-, x)
gradient(::typeof(+), grad, output, x) = ( grad * one(x),)
gradient(::typeof(-), grad, output, x) = (-grad * one(x),)
+(x::Node, y::Node) = register(+, x, y)
-(x::Node, y::Node) = register(-, x, y)
*(x::Node, y::Node) = register(*, x, y)
/(x::Node, y::Node) = register(/, x, y)
gradient(::typeof(+), grad, output, x, y) = grad * one(x),   grad * one(y)
gradient(::typeof(-), grad, output, x, y) = grad * one(x),   grad *-one(y)
gradient(::typeof(*), grad, output, x, y) = grad * y,        grad * x
gradient(::typeof(/), grad, output, x, y) = grad * one(x)/y, grad *-x / y / y

gradient(::Broadcasted{typeof(+)}, grad, output, x, y) = grad .* one.(x),   grad .* one.(y)
gradient(::Broadcasted{typeof(-)}, grad, output, x, y) = grad .* one.(x),   grad .*-one.(y)
gradient(::Broadcasted{typeof(*)}, grad, output, x, y) = grad .* y,         grad .* x
gradient(::Broadcasted{typeof(/)}, grad, output, x, y) = grad .* one.(x)./y, -grad .* x ./ y ./ y

import Base: abs, sin, cos, tan, exp, log, sqrt, max
abs(x::Node)  = register(abs, x)
sin(x::Node)  = register(sin, x)
cos(x::Node)  = register(cos, x)
tan(x::Node)  = register(tan, x)
exp(x::Node)  = register(exp, x)
log(x::Node)  = register(log, x)
sqrt(x::Node) = register(sqrt, x)
max(x::Node, y::Node) = register(max, isless(value(x), value(y)) ? y : x)
gradient(::typeof(abs), grad, output, x)  = (grad * sign(x),)
gradient(::typeof(sin), grad, output, x)  = (grad * cos(x),)
gradient(::typeof(cos), grad, output, x)  = (grad *-sin(x),)
gradient(::typeof(tan), grad, output, x)  = (grad *(tan(x)^2 + 1),)
gradient(::typeof(exp), grad, output, x)  = (grad * exp(x),)
gradient(::typeof(log), grad, output, x)  = (grad / x,)
gradient(::typeof(sqrt), grad, output, x) = (grad * 0.5/sqrt(x),)
gradient(::typeof(max), grad, output, x)  = (grad * one(x),)

gradient(::Broadcasted{typeof(sin)}, grad, output, x) = (grad .* cos.(x),)
gradient(::Broadcasted{typeof(cos)}, grad, output, x) = (grad .*-sin.(x),)
gradient(::Broadcasted{typeof(tan)}, grad, output, x) = (grad .*(tan.(x).^2 + 1),)
gradient(::Broadcasted{typeof(exp)}, grad, output, x) = (grad .* exp.(x),)
gradient(::Broadcasted{typeof(log)}, grad, output, x) = (grad ./ x,)
gradient(::Broadcasted{typeof(sqrt)}, grad, output, x) = (grad .* 0.5 ./ sqrt.(x),)

import Base: length, eltype
length(x::Node) = length(value(x))
eltype(x::Node) = eltype(value(x))

import Base: transpose, sum, vec
transpose(x::Node) = register(transpose, x)
maxpool(x::Node) = register(maxpool, x)
conv(k::Node, x::Node, n, m) = register(conv, k, x, n, m)

gradient(::typeof(transpose), grad, output, x::AbstractMatrix) = (transpose(grad),)
gradient(::typeof(maxpool), grad, output, input) = begin
    n, m, K = size(input)
    N, M = floor.(Integer, (n, m) ./ 2)
    dinput = zero(input)
    for i=1:N
        for j=1:M
            matched = zeros(2, 2, K);
            dregion = @view dinput[ 2(i)-1:2(i)-0, 2(j)-1:2(j)-0, :]
            for k=1:K
                region = @view input[ 2(i)-1:2(i)-0, 2(j)-1:2(j)-0, k]
                matched[:,:,k] = grad[i,j,k] * (region .== output[i, j, k])
            end
            dregion.+= matched
        end
    end

    (dinput,)
end
gradient(::typeof(conv), grad, output, kernel, input, n, m) = begin
    kernel = reshape(kernel, 3, 3, :) # 3×3×2
    input  = reshape(input, n, m) # 6×6
    dkernel = zero(kernel) # 3×3×2
    dinput = zero(input) # 6×6
    N, M = (n, m) .- 2
    _, _, K = size(kernel)
    for i=1:N # 4
        for j=1:M # 4
            for k=1:K # 2
                region = input[i:i+2, j:j+2] # 3×3
                dkernel[:, :, k] += region .* grad[i,j,k]
                dinput[i:i+2, j:j+2] += kernel[:, :, k] .* grad[i,j,k]
            end
        end
    end
    dkernel, dinput
end
gradient(::typeof(*), grad, output, lhs::AbstractArray, rhs::AbstractArray) =
    grad * transpose(rhs), transpose(lhs) * grad
gradient(::typeof(+), grad, output, x::AbstractArray) = ( grad .* one.(x),)
gradient(::typeof(-), grad, output, x::AbstractArray) = (-grad .* one.(x),)

import LinearAlgebra: dot
dot(x::Node, y::Node) = register(dot, x, y)
gradient(::typeof(dot), grad, output, x, y) =
    grad .* y, grad .* x

import Base.Broadcast: broadcastable, broadcasted, materialize
struct ComputGraphStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:Node}) = ComputGraphStyle()
Broadcast.BroadcastStyle(cgs::ComputGraphStyle, ::Broadcast.BroadcastStyle) = cgs
forward(op::Broadcasted, args...) = broadcasted(op.f, args...)
broadcastable(x::Node) = x
broadcasted(::ComputGraphStyle, f, args...) = register(Broadcasted(f), args...)
materialize(x::Node) = register(materialize, x)

function backward(cached::CachedNode, ::typeof(materialize), grad)
    backward(arg(cached, 1), grad)
end

+(x::Number, y::Tensor) = broadcast(+, x, y)
+(x::Number, y::TensorCachedNode) = broadcast(+, x, y)
+(x::Tensor, y::Number) = broadcast(+, x, y)
+(x::TensorCachedNode, y::Number) = broadcast(+, x, y)

-(x::Number, y::Tensor) = broadcast(-, x, y)
-(x::Number, y::TensorCachedNode) = broadcast(-, x, y)
-(x::Tensor, y::Number) = broadcast(-, x, y)
-(x::TensorCachedNode, y::Number) = broadcast(-, x, y)

*(x::Number, y::Tensor) = broadcast(*, x, y)
*(x::Number, y::TensorCachedNode) = broadcast(*, x, y)
*(x::Tensor, y::Number) = broadcast(*, x, y)
*(x::TensorCachedNode, y::Number) = broadcast(*, x, y)

/(x::Number, y::Tensor) = broadcast(/, x, y)
/(x::Number, y::TensorCachedNode) = broadcast(/, x, y)
/(x::Tensor, y::Number) = broadcast(/, x, y)
/(x::TensorCachedNode, y::Number) = broadcast(/, x, y)

import Base: iterate
iterate(x::Node) = iterate_forward(iterate(value(x)), x)
iterate(x::Node, st) = iterate_forward(iterate(value(x), st), x, st)
iterate_forward(out::Nothing, x::Node) = nothing
iterate_forward(out::Nothing, x::Node, st) = nothing

function iterate_forward(out, x::Node, st)
    node = ComputableNode(iterate, (x, st))
    v, st = out
    CachedNode(node, v), st
end

function iterate_forward(out, x::Node)
    node = ComputableNode(iterate, (x, ))
    v, st = out
    CachedNode(node, v), st
end

function gradient(::typeof(iterate), grad::Number, output, x::AbstractArray)
    out_grad = zero(x)
    out_grad[1] = grad
    (out_grad, )
end

function gradient(::typeof(iterate), grad, output, x::AbstractArray, st)
    out_grad = zero(x)
    out_grad[st] = grad
    (out_grad, )
end

sum(x::Node; dims=:) = register(sum, x; dims=dims)
gradient(::typeof(sum), grad, output, x; kwargs...) = (grad .* one.(x),)

import Base: getindex, selectdim, view, reshape

getindex(x::Node, inds...) = register(getindex, x, inds...)
gradient(::typeof(getindex), grad, output, x::Number, ind::Int) = (grad, )
function gradient(::typeof(getindex), grad, output, x::AbstractArray, inds...)
    grad_output = fill!(similar(x), 0)
    setindex!(grad_output, grad, inds...)
    (grad_output, )
end

selectdim(x::Node, d, i) = register(selectdim, x, d, i)
function gradient(::typeof(selectdim), grad, output, x::AbstractArray, d, i)
    grad_output = fill!(similar(x), 0)
    subgrad = selectdim(grad_output, d, i)
    setindex!(subgrad, grad, :)
    (grad_output, )
end

view(x::Node, inds...) = register(view, x, inds...)
function gradient(::typeof(view), grad, output, x::AbstractArray, inds...)
    grad_output = fill!(similar(x), 0)
    subgrad = view(grad_output, inds...)
    setindex!(subgrad, grad, :)
    (grad_output, )
end

reshape(x::Node, dims...) = register(reshape, x, dims...)
gradient(::typeof(reshape), grad, output, x, dims...) = (reshape(grad, size(x)...),)
