import LinearAlgebra: diagm
Float = Float64
abstract type Layer end
abstract type ActivationFunction end
abstract type sigmoid <: ActivationFunction end
abstract type softmax <: ActivationFunction end
abstract type linear  <: ActivationFunction end
abstract type ReLU    <: ActivationFunction end
abstract type tanh    <: ActivationFunction end
abstract type LossFunction end
abstract type mean_squared_loss <: LossFunction end
abstract type cross_entropy_loss <: LossFunction end

struct FullyConnectedLayer{T <: ActivationFunction} <: Layer
  w  :: AbstractMatrix{Float} # weights
  b  :: AbstractMatrix{Float} # bias
  u  :: Matrix{Float} # input
  ū  :: Matrix{Float} # sum
  û  :: Matrix{Float} # output
  ūb :: Matrix{Float} # derivatives...
  ūw :: Matrix{Float}
  ūu :: Matrix{Float}
  ûū :: Matrix{Float}
  Ew :: AbstractMatrix{Float}
  Eb :: AbstractMatrix{Float}
end
function FullyConnectedLayer{T}(w, b, Ew, Eb) where T
  n, m = size(w)

  FullyConnectedLayer{T}(w, b,
	zeros(m,1), zeros(n,1), zeros(n,1),
	zeros(n,1), zeros(1,m), zeros(m,n), zeros(n,n),
	Ew, Eb)
end

struct NeuralNetwork{T <: LossFunction}
  hidden :: Layer
  output :: Layer
end

loss(nn::NeuralNetwork{mean_squared_loss}, y, ŷ)  = sum(0.5(y - ŷ).^2)
loss(nn::NeuralNetwork{cross_entropy_loss}, y, ŷ) = sum(-y .* log.(ŷ))

function loss(nn::NeuralNetwork{T}, inputs, targets, ks) where T
  Eₜ  = 0.
  for k = ks
    x   = reshape( inputs[k,:], :, 1)
    y   = reshape(targets[k,:], :, 1)
    ŷ   = feedforward!(nn, x)
    Eₜ += loss(nn, y, ŷ)
  end
  return Eₜ/length(ks)
end

∂loss(nn::NeuralNetwork{mean_squared_loss}, y, ŷ)  = -(y - ŷ)
∂loss(nn::NeuralNetwork{cross_entropy_loss}, y, ŷ) = -y ./ ŷ

function feedforward!(fc::FullyConnectedLayer{T}, u) where T
  fc.u .= u
  fc.ū .= fc.w * u + fc.b
  fc.û .= activation(fc)
end

function feedforward!(nn::NeuralNetwork, x)
  x̂ = feedforward!(nn.hidden, x)
  ŷ = feedforward!(nn.output, x̂)
end


activation(fc::FullyConnectedLayer{sigmoid})= (@. 1.0 / (1.0 + exp(-1.0fc.ū)))
activation(fc::FullyConnectedLayer{tanh})   = (@. 2.0 / (1.0 + exp(-2.0fc.ū)) - 1.0)
activation(fc::FullyConnectedLayer{linear}) =  fc.ū
activation(fc::FullyConnectedLayer{ReLU})   = (@. max(0, fc.ū))
activation(fc::FullyConnectedLayer{softmax})=  exp.(fc.ū) ./ sum(exp.(fc.ū))

eye(v::Matrix{Float}) = Matrix(1.0I, size(v))
diagonal(v::Matrix{Float}) = diagm(0 => vec(v))
∂activation(fc::FullyConnectedLayer{sigmoid})= (@. fc.û * (1.0 - fc.û)) |> diagonal
∂activation(fc::FullyConnectedLayer{tanh})   = (@. 1.0 - fc.û^2) |> diagonal
∂activation(fc::FullyConnectedLayer{linear}) =  fc.ûū |> eye
∂activation(fc::FullyConnectedLayer{ReLU})   =  fc.û |> diagonal .|> (ûi) -> ûi > 0. ? 1. : 0.
∂activation(fc::FullyConnectedLayer{softmax})= (fc.û |> diagonal).- fc.û * (fc.û |> transpose)


function backpropagate!(fc::FullyConnectedLayer{T}) where T
  fc.ūb .= ê = ones(size(fc.ūb))
  fc.ūw .= fc.u |> transpose
  fc.ūu .= fc.w |> transpose
  fc.ûū .= ∂activation(fc)
end

function backpropagate!(nn::NeuralNetwork{T}, y) where T
  backpropagate!(nn.output)
  backpropagate!(nn.hidden)
  ŷ = nn.output.û
  
  ŷȳ = nn.output.ûū
  ȳx̂ = nn.output.ūu
  x̂x̄ = nn.hidden.ûū

  Eŷ  = ∂loss(nn, y, ŷ)
  Eȳ  = ŷȳ * Eŷ
  Ex̂  = ȳx̂ * Eȳ
  Ex̄  = x̂x̄ * Ex̂
  
  ȳwₒ = nn.output.ūw
  ȳbₒ = nn.output.ūb
  x̄wₕ = nn.hidden.ūw
  x̄bₕ = nn.hidden.ūb
  nn.output.Ew .= Eȳ * ȳwₒ
  nn.output.Eb .= Eȳ.* ȳbₒ
  nn.hidden.Ew .= Ex̄ * x̄wₕ
  nn.hidden.Eb .= Ex̄.* x̄bₕ
end
