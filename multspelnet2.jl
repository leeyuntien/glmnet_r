module multspelnet2

using CSV, DataFrames, PyCall, Revise
using Statistics, BenchmarkTools
using Distributed

NUM_WORKER = 8
addprocs(NUM_WORKER)

@everywhere using SparseArrays, SharedArrays, LinearAlgebra

@everywhere struct glmnet_param
    α
    λ
    max_iter
    ϵ
    minλ
    θ
    mnλ
    big
    standardize
    intercept
end

ind_data = #load_npz("")
dep_data = #load_npz("")
x = sparse(SparseMatrixCSC(maximum(ind_data.indices) + 1, length(ind_data.indptr) - 1, ind_data.indptr .+ 1, ind_data.indices .+ 1, ind_data.data)')
y = Array(SparseMatrixCSC(maximum(dep_data.indices) + 1, length(dep_data.indptr) - 1, dep_data.indptr .+ 1, dep_data.indices .+ 1, dep_data.data)')

@everywhere x = $x
@everywhere y = $y

n_samples = x.m
n_features = x.n
n_targets = size(y)[2]
if x.m != size(y)[1]
    @show "Dimens not match!!"
end

glmparams = glmnet_param(0.9, [0.01], 100, 1e-6, 1e-4, 1e-7, 5, 9.9e35, false, true)
nλ = length(glmparams.λ)
coeffs = zeros(n_targets, n_features) # final coefficients
w = ones(n_samples) # weights on samples
o = zeros(n_targets) # mean shift
w ./= sum(w) # default
#ix = zeros(n_features)
#ju = zeros(n_features)

@everywhere glmparams = $glmparams

# temporary variables
alm = 0.0
alf = 1.0
iz = 0
jz = 1
nin = 0
flmin = 1.0 # less than 1.0 only if no lambda so assume always provide lambda
del = zeros(n_targets)
mm = zeros(n_features)
ia = zeros(n_features)
gj = zeros(n_targets)
gk = zeros(n_targets)
g = zeros(n_features)
xm = zeros(n_features) # mean of x
xs = zeros(n_features) # std of x
xv = zeros(n_features) # var of x
ym = zeros(n_targets) # mean of y
ys = zeros(n_targets) # std of y
vp = ones(n_features) # penalty factor
ys0 = preprocess(glmparams.intercept, glmparams.standardize) # side effects to some variables
rsq = ys0
thr = glmparams.θ
thr *= ys0 / n_targets
if (flmin < 1.0) # assume no use
    eqs = max(glmparams.ϵ, flmin)
    alf = eqs^(1.0 / (nλ - 1))
end
for j in 1:n_features
    #if (!ju[j])
    #    continue
    #end
    gj .= y' * (x[:, j] .* w)
    g[j] = norm(gj) / xs[j]
end

gj_shared = SharedArray{Float64}(n_targets, NUM_WORKER)
gk_shared = SharedArray{Float64}(n_targets, NUM_WORKER)
gkn_shared = SharedArray{Float64}(NUM_WORKER)
u_shared = SharedArray{Float64}(NUM_WORKER)
del_shared = SharedArray{Float64}(n_targets, NUM_WORKER)
o_shared = SharedArray{Float64}(n_targets)
o_min = copy(o)
dlx_shared = SharedArray{Float64}(1)
y_shared = SharedArray{Float64}(n_samples, n_targets)
y_min = copy(y)
coeffs_shared = SharedArray{Float64}(n_targets, n_features)
coeffs_min = copy(coeffs)

y_shared .= y
dlx_min = 1e6

nlp = 0 # current iteration counter
for m in 1:nλ
    alm0 = alm
    if (flmin >= 1.0)
        alm = glmparams.λ[m]
    elseif (m > 2) # assume no use
        alm *= alf
    elseif (m == 1) # assume no use
        alm = glmparams.big
    else # assume no use
        alm = 0.0
        for j in 1:n_features
            if (!ju[j])
                continue
            end
            if (vp[j] > 0.0)
                alm0 = max(alm0, g[j] / vp[j])
            end
        end
        alm0 /= max(alpha, 1e-3)
        alm = alf * alm0
    end

    dem = alm * (1.0 - glmparams.α)
    ab = alm * glmparams.α
    @everywhere dem = $dem
    @everywhere ab = $ab
    rsq0 = rsq
    jz = 1
    #tlam = glmparams.α * (2.0 * alm - alm0)
    #for k in 1:n_features
    #    if (ix[k] || !ju[k])
    #        continue
    #    end
    #    if (g[k] > tlam * vp[k])
    #        ix[k] = 1
    #    end
    #end

    if (iz * jz != 0)
        multspelnet2_do_b()
    end

    while true
        converged_kkt = false
        while true
            if (nlp > glmparams.max_iter)
                @show "maximum iteration reached"
                break
            end
            nlp += 1
            dlx_shared[1] = 0.0
            @time @sync @distributed for k in 1:n_features
                if sum(x[:, k]) < glmparams.ϵ
                    continue
                end
                
                n = myid() - 1
                for j in 1:n_targets
                    gj_shared[j, n] = dot(x[:, k] .* w, y_shared[:, j] .+ o_shared[j]) / xs[k]
                    gk_shared[j, n] = gj_shared[j, n] + xv[k] * coeffs_shared[j, k]
                end
                gkn_shared[n] = norm(gk_shared[:, n])
                u_shared[n] = 1.0 - ab * sqrt(xv[k]) / gkn_shared[n]
                del_shared[:, n] .= coeffs_shared[:, k]
                if (u_shared[n] <= 0.0)
                    coeffs_shared[:, k] .= 0.0
                else
                    coeffs_shared[:, k] .= gk_shared[:, n] .* (u_shared[n] / (xv[k] * (1.0 + dem))) # omit check bounds
                end
                #del_shared[:, n] .= coeffs_shared[:, k] .- del_shared[:, n]
                del_shared[:, n] .*= -1.0
                del_shared[:, n] .+= coeffs_shared[:, k]

                if (maximum(abs.(del_shared[:, n])) > 0.0)
                    y_shared .-= del_shared[:, n]' ./ xs[k] .* x[:, k]
                    o_shared .+= (xm[k] / xs[k]) .* del_shared[:, n] # have to do in proper order
                    dlx_shared[1] = max(dlx_shared[1], xv[k] * maximum(del_shared[:, n] .* del_shared[:, n])) # possible in bulk
                end
            end
            @show nlp, dlx_shared[1]
            if dlx_shared[1] < dlx_min
                dlx_min = dlx_shared[1]
                y_min .= y_shared
                o_min .= o_shared
                coeffs_min .= coeffs_shared
            end

            if (dlx_shared[1] < thr)
                ixx = false
                if ixx
                    continue
                end
                converged_kkt = true
            end
            break
        end

        if (converged_kkt)
            break
        end
        if (nlp > glmparams.max_iter)
            @show "maximum iteration reached"
            break
        end

    end
end

end
