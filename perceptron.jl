import LinearAlgebra
import Random
Random.seed!(0)

diagonal(v) = LinearAlgebra.diagm(0 => vec(v))
sigmoid(x :: Real) = one(x) / (one(x) + exp(-x))
softmax(x :: Array{<:Real,2}) = exp.(x) ./ sum(exp.(x))
linear(x :: Real)  = x
ReLU(x :: Real) = max(0, x)
σ = sigmoid
∑ = sum

η = 0.1 # learning rate
epochs = 5000

batch_size     = 10
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
for i=1:epochs
  global wₕ, wₒ, bₕ, bₒ
  ∂E_∂wₒ = zeros(size(wₒ))
  ∂E_∂wₕ = zeros(size(wₕ))
  ∂E_∂bₒ = zeros(size(bₒ))
  ∂E_∂bₕ = zeros(size(bₕ))

for _=1:batch_size
  j = rand(1:size(inputs, 1))
  x = reshape( inputs[j,:], :, 1)
  y = reshape(targets[j,:], :, 1)

  ### Feed forward ###
  x̄ = wₕ * x .+ bₕ
  x̂ = ReLU.(x̄)
  x̂ᵀ = transpose(x̂)
  
  ȳ = wₒ * x̂ .+ bₒ
  ŷ = softmax(ȳ)
  ŷᵀ = transpose(ŷ)
  
  E = -y .* log.(ŷ)

  ### Back propagation ###
  ∂ȳ_∂bₒ = eₒ
  ∂ȳ_∂wₒ = x̂ᵀ
  ∂ŷ_∂ȳ  = -(ŷ * ŷᵀ) .+ (ŷ |> diagonal)
  ∂E_∂ŷ  = -y ./ ŷ
  
  ∂E_∂ȳ  = ∂ŷ_∂ȳ * ∂E_∂ŷ

  ∂E_∂wₒ+= ∂E_∂ȳ * ∂ȳ_∂wₒ
  ∂E_∂bₒ+= ∂E_∂ȳ.* ∂ȳ_∂bₒ
  
  ∂x̄_∂bₕ = eₕ
  ∂x̄_∂wₕ = x  |> transpose
  ∂x̂_∂x̄  = x̂ .|> (x̂i) -> x̂i > 0. ? 1. : 0.
  ∂ȳ_∂x̂  = wₒ |> transpose

  ∂E_∂x̂  = ∂ȳ_∂x̂ * ∂E_∂ȳ
  ∂E_∂x̄  = ∂E_∂x̂.* ∂x̂_∂x̄

  ∂E_∂wₕ+= ∂E_∂x̄ * ∂x̄_∂wₕ
  ∂E_∂bₕ+= ∂E_∂x̄.* ∂x̄_∂bₕ
  println(i, "\t", sum(E))
end
  wₒ    -= η * ∂E_∂wₒ/batch_size
  wₕ    -= η * ∂E_∂wₕ/batch_size
  bₒ    -= η * ∂E_∂bₒ/batch_size
  bₕ    -= η * ∂E_∂bₕ/batch_size
end
