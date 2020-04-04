function subviews(u, ns...)
   vs, s = [], 0
   for n in ns
     v = view(u, (1:n) .+ s)
     s += n
     push!(vs, v)
   end    
   vs   
end 