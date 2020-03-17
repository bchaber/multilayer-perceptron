using LinearAlgebra
abstract type DescentMethod end

struct GradientDescent <: DescentMethod
  α
end

mutable struct BFGS <: DescentMethod
  Q
end
BFGS(n::Integer) = BFGS(Matrix(1.0I, n, n))

mutable struct Momentum <: DescentMethod
  α # learning rate
  β # momentum decay
  v # momentum
end
Momentum(α, β, n::Integer) = Momentum(α, β, zeros(n))

mutable struct Adagrad <: DescentMethod
  α # learning rate
  ε # small value
  s # sum of squared gradient
end
Adagrad(α, ε, n::Integer) = Adagrad(α, ε, zeros(n))

mutable struct RMSProp <: DescentMethod
  α # learning rate
  γ # decay
  ε # small value
  s # sum of squared gradient
end
RMSProp(α, γ, ε, n::Integer) = RMSProp(α, γ, ε, zeros(n))

function step!(M::RMSProp, f, ∇f, x)
  α, γ, ε, s, g = M.α, M.γ, M.ε, M.s, ∇f(x)
  s[:] = γ*s + (1-γ)*(g.*g)
  return x - α*g ./ (sqrt.(s) .+ ε)
end

function step!(M::Adagrad, f, ∇f, x)
  α, ε, s, g = M.α, M.ε, M.s, ∇f(x)
  s[:] += g.*g
  return x - α*g ./ (sqrt.(s) .+ ε)
end

function step!(M::Momentum, f, ∇f, x) 
  α, β, v, g = M.α, M.β, M.v, ∇f(x)
  v[:] = β*v .- α*g
  return x + v
end

function step!(M::GradientDescent, f, ∇f, x)
  α, g = M.α, ∇f(x)
  return x - α*g
end

function bracket_minimum(f, x=0; s=1e-2, k=2.0)
  a, ya = x, f(x)
  b, yb = a + s, f(a + s)
  if yb > ya
    a, b = b, a
    ya, yb = yb, ya
    s = -s
  end
  while true
    c, yc = b + s, f(b + s)
    if yc > yb
      return a < c ? (a, c) : (c, a)
    end
    a, ya, b, yb = b, yb, c, yc
    s *= k
  end
end

function fibonacci_search(f, a, b, n; ε=0.01)
  s = (1 - √5)/(1 + √5)
  ρ = 1 / (φ*(1-s^(n+1))/(1-s^n))
  d = ρ*b + (1-ρ)*a
  yd = f(d)
  for i in 1:n-1
    if i == n-1
      c = ε*a + (1-ε)*d
    else
      c = ρ*a + (1-ρ)*b
    end
    yc = f(c)
    if yc < yd
      b, d, yd = d, c, yc
    else
      a, b = b, c
    end
    ρ = 1 / (φ*(1-s^(n-i+1))/(1-s^(n-i)))
  end
  return a < b ? (a, b) : (b, a)
end

function golden_section_search(f, a, b, n)
  ρ = φ - 1
  d = ρ * b + (1 - ρ)*a
  yd = f(d)
  for i = 1:n-1
    c = ρ*a + (1 - ρ)*b
    yc = f(c)
    if yc < yd
      b, d, yd = d, c, yc
    else
      a, b = b, c
    end
  end
  return a < b ? (a, b) : (b, a)
end

function minimize(objective, a, b)
  x, y = golden_section_search(objective, a, b, 50)
  x/2 + y/2
end

function line_search(f, x, d)
  objective = α -> f(x + α*d)
  a, b = bracket_minimum(objective)
  α = minimize(objective, a, b)
  return x + α*d
end

function step!(M::BFGS, f, ∇f, x)
  Q, g = M.Q, ∇f(x)
  x′ = line_search(f, x, -Q*g)
  g′ = ∇f(x′)
  δ = x′ - x
  γ = g′ - g
  Q[:] = Q - (δ*γ'*Q + Q*γ*δ')/(δ'*γ) + (1 + (γ'*Q*γ)/(δ'*γ))[1]*(δ*δ')/(δ'*γ)
  return x′
end