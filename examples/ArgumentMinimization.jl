include("../models/random.jl")
include("../solver/solver.jl")


model=randomModel(2,[10,10,10,10,10])
x0=rand(Uniform(-20,20),model.n0)
(code,trace1,x)=solverLP(model,x0);

# using Plots
# env=4.25
# xrange=[x[1]-env,x[1]+env]
# yrange=[x[2]-env,x[2]+env]
# #= plotF(xrange,yrange) =#
# #= function plotF(xrange,yrange) =#
# res = 50
# xvals = range(xrange[1],stop=xrange[2], length=res)
# yvals = range(yrange[1],stop=yrange[2], length=res)
# #= y = x =#
# # z=rand(res,res)
# z = Array{Float64}(undef,res, res);
# for i in 1:res
#     for j in 1:res
# 		z[j, i] = (f(model,[xvals[i], yvals[j]]))[1];
#     end
# end
# Plots.surface(xvals, yvals, z',legend=false, axis=[],framestyle=:none)
#
# xsteps=[trace1[i][1].x for i in 1:length(trace1)]
# plot(first.(xsteps), last.(xsteps))
#
#
#
# model=randomModel(1,[50,10,10,10,10,10])
#
# x0=[0.0]
# (code,trace1,x)=solverLP(model,x0);
#
# xopt=x[1]
# xrange=LinRange(xopt-0.8,xopt+0.8,1000)
# yvals = [f(model,x)[1] for x in xrange]
# xsteps=[trace1[i][1].x[1] for i in 1:length(trace1)]
# Plots.plot(xrange,yvals, label="target function")
# vline!(xsteps,label="iteration steps")
# vline!([x],label="final position",width=1.5)
#
#
#
