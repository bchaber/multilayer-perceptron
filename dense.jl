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

function feedforward!(fc::FullyConnectedLayer{T}, u) where T
  fc.u .= u
  fc.ū .= fc.w * u + fc.b
  fc.û .= activation(fc)
end

function backpropagate!(fc::FullyConnectedLayer{T}) where T
  fc.ūb .= ê = ones(size(fc.ūb))
  fc.ūw .= fc.u |> transpose
  fc.ūu .= fc.w |> transpose
  fc.ûū .= ∂activation(fc)
end

activation(fc::FullyConnectedLayer{sigmoid})= (@. 1.0 / (1.0 + exp(-1.0fc.ū)))
activation(fc::FullyConnectedLayer{tanh})   = (@. 2.0 / (1.0 + exp(-2.0fc.ū)) - 1.0)
activation(fc::FullyConnectedLayer{linear}) =  fc.ū
activation(fc::FullyConnectedLayer{ReLU})   = (@. max(0, fc.ū))
activation(fc::FullyConnectedLayer{softmax})=  exp.(fc.ū) ./ sum(exp.(fc.ū))

eye(n::Integer) = Matrix(1.0I, n, n)
diagonal(v::Matrix{Float}) = diagm(0 => vec(v))
∂activation(fc::FullyConnectedLayer{sigmoid})= (@. fc.û * (1.0 - fc.û)) |> diagonal
∂activation(fc::FullyConnectedLayer{tanh})   = (@. 1.0 - fc.û^2) |> diagonal
∂activation(fc::FullyConnectedLayer{linear}) =  fc.û |> length   |> eye
∂activation(fc::FullyConnectedLayer{ReLU})   =  fc.û |> diagonal .|> (ûi) -> ûi > 0. ? 1. : 0.
∂activation(fc::FullyConnectedLayer{softmax})= (fc.û |> diagonal).- fc.û * (fc.û |> transpose)