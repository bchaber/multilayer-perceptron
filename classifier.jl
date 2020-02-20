import LinearAlgebra: diagm
import Random: shuffle, seed!
seed!(0)

diagonal(v) = diagm(0 => vec(v))
sigmoid(x :: Real) = one(x) / (one(x) + exp(-x))
softmax(x :: Array{<:Real,2}) = exp.(x) ./ sum(exp.(x))
linear(x :: Real)  = x
ReLU(x :: Real) = max(0, x)

η = 0.001 # learning rate
epochs = 500

batch_size     = 1
input_neurons  = 4
hidden_neurons = 8
output_neurons = 3

e  = ones(input_neurons,  1)
eₕ = ones(hidden_neurons, 1)
eₒ = ones(output_neurons, 1)
bₕ = zeros(hidden_neurons, 1)
bₒ = zeros(output_neurons, 1)
wₕ = randn(hidden_neurons, input_neurons)
wₒ = randn(output_neurons, hidden_neurons)
include("iris.jl")

test_size  =  10
train_size = 140
data_size  = train_size + test_size
train_set  = shuffle(1:data_size)[1:train_size]
test_set   = setdiff(1:data_size, train_set)

function feedforward(x, wₕ, bₕ, wₒ, bₒ)
  x̄ = wₕ * x .+ bₕ
  x̂ = ReLU.(x̄)
  
  ȳ = wₒ * x̂ .+ bₒ
  ŷ = softmax(ȳ)

  return ŷ, ȳ, x̂, x̄
end

function backpropagate(ŷ, ȳ, y, x̂, x̄, x)
  global eₒ, eₕ
  xᵀ,wₒᵀ = x |> transpose, wₒ|> transpose
  x̂ᵀ, ŷᵀ = x̂ |> transpose, ŷ |> transpose
  ∂ȳ_∂bₒ = eₒ
  ∂ȳ_∂wₒ = x̂ᵀ
  ∂ŷ_∂ȳ  = -(ŷ * ŷᵀ) .+ (ŷ |> diagonal)
  ∂E_∂ŷ  = -y ./ ŷ
  
  ∂E_∂ȳ  = ∂ŷ_∂ȳ * ∂E_∂ŷ

  ∂E_∂wₒ = ∂E_∂ȳ * ∂ȳ_∂wₒ
  ∂E_∂bₒ = ∂E_∂ȳ.* ∂ȳ_∂bₒ
  
  ∂x̄_∂bₕ = eₕ
  ∂x̄_∂wₕ = xᵀ
  ∂x̂_∂x̄  = x̂ .|> (x̂i) -> x̂i > 0. ? 1. : 0.
  ∂ȳ_∂x̂  = wₒᵀ

  ∂E_∂x̂  = ∂ȳ_∂x̂ * ∂E_∂ȳ
  ∂E_∂x̄  = ∂E_∂x̂.* ∂x̂_∂x̄

  ∂E_∂wₕ = ∂E_∂x̄ * ∂x̄_∂wₕ
  ∂E_∂bₕ = ∂E_∂x̄.* ∂x̄_∂bₕ

  return ∂E_∂wₒ, ∂E_∂bₒ, ∂E_∂wₕ, ∂E_∂bₕ
end

for i=1:epochs
  global wₕ, wₒ, bₕ, bₒ
  ∂wₒ = zeros(size(wₒ))
  ∂wₕ = zeros(size(wₕ))
  ∂bₒ = zeros(size(bₒ))
  ∂bₕ = zeros(size(bₕ))

  for _=1:batch_size
  j = rand(train_set)
  x = reshape( inputs[j,:], :, 1)
  y = reshape(targets[j,:], :, 1)

  ŷ, ȳ, x̂, x̄ =
  feedforward(x, wₕ, bₕ, wₒ, bₒ)
  Eₜ  = sum(-y .* log.(ŷ))
  ∂E_∂wₒ, ∂E_∂bₒ, ∂E_∂wₕ, ∂E_∂bₕ =
  backpropagate(ŷ, ȳ, y, x̂, x̄, x)
  ∂wₒ += ∂E_∂wₒ
  ∂bₒ += ∂E_∂bₒ
  ∂wₕ += ∂E_∂wₕ
  ∂bₕ += ∂E_∂bₕ
  end

  Eₜ  = 0.
  for k = test_set
    x = reshape( inputs[k,:], :, 1)
    y = reshape(targets[k,:], :, 1)
    ŷ, ȳ, x̂, x̄ =
    feedforward(x, wₕ, bₕ, wₒ, bₒ)
    Eₜ += sum(-y .* log.(ŷ))
  end
  println(i, "\t", Eₜ/test_size)
  wₒ    -= η * ∂wₒ/batch_size
  wₕ    -= η * ∂wₕ/batch_size
  bₒ    -= η * ∂bₒ/batch_size
  bₕ    -= η * ∂bₕ/batch_size
end