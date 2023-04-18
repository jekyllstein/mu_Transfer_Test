### A Pluto.jl notebook ###
# v0.19.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ cb90ded3-e7d8-4af3-ade7-6794317c3f93
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(mktempdir())
	Pkg.add(url = "https://github.com/Blackbody-Research/NVIDIALibraries.jl")
	Pkg.add(url = "https://github.com/Blackbody-Research/FCANN.jl", rev = "batch-records")
	pkglist = ["Random", "Statistics", "StatsBase", "PlutoPlotly", "PlutoUI", "JSON", "JSON3"]
	Pkg.add(pkglist)

	using FCANN, Random, Statistics, StatsBase, PlutoPlotly, PlutoUI, JSON, JSON3 
	
	setBackend(:GPU)
	getBackend() == :GPU && switch_device(3)

	#defining this inside here ensures that the results dictionary is loaded from the start
	function readresultsjson!(fname, resultsdict)
		dict1 = JSON.parsefile(fname)
		convertkey(key) = eval(Meta.parse(key))
		#this is parsed as a vector of vectors intead of a matrix
		function convertoutput(a)
			Symbol(a[1]) == :costrecord && return :costrecord => Float32.(a[2])
			Symbol(a[1]) => UInt8.(a[2])
		end
		convertvalue(value) = NamedTuple(convertoutput(a) for a in value)
		for a in dict1
			resultsdict[convertkey(a[1])] = convertvalue(a[2])
		end
	end

	# load previous results if they exist
	global trainresultsdict = Dict{NamedTuple, NamedTuple}()
	if isfile("trainresults.json")
		readresultsjson!("trainresults.json", trainresultsdict)
	end

	using Base.Threads
	using LinearAlgebra.BLAS
	PlutoUI.TableOfContents()
end

# ╔═╡ 33300f8c-ba16-11ed-08bd-6b6cdd6a420d
md"""
# Introduction

In the paper [`Tensor Programs V:Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer`](https://arxiv.org/pdf/2203.03466.pdf), the authors claim to have a method of parameter initialization and learning rate tuning that can be copied from a small model to a large model.  The method is called *Maximal Update Parametrization (abbreviated μP)*.  The proposed algorithm functions as follows:
1. Parametrize a target model with μP
2. Tune a smaller version of the model in width, depth, or both on parameters such as learning rate, learning rate schedule, etc...
3. Copy tuned parameters to target model.

μP parametrization includes a per layer initialization noise variance as well as a per layer learning rate.  

## Initialization
Consider first an MLP with LeCun initialization with hidden layers of size n.  In this case we initialize the weights as follows: $W^1 \sim \mathcal{N}(0, 1/d_{in})$, $W^{i>1} \sim \mathcal{N}(0, 1/n)$, $b^i = 0$.  So each weight matrix initialization variance is scaled by the inverse of the input dimension.

With μP all weight initializations would be the same except for the output weights:
$W^{output} \sim \mathcal{N}(0, 1/n^2)$.  In general the difference is that the output weights are scaled by $1 / \text{fan in}^2$.

## Learning Rate
Using ADAmax, consider a base learning rate η.  The learning rate multiplication factor used for all biases and input weights should be 1.  For the output and hidden weights the factor is 1/fan_in.
"""

# ╔═╡ 1b8394bd-5dd4-4229-ac1f-507961f4b527
md"""
# Functions
"""

# ╔═╡ f9f406cd-6082-4bac-8f34-53b7527fd002
md"""
## Create Data
"""

# ╔═╡ 076086e4-12b4-4a7d-bbca-3bd54072ef07
e(k, x) = exp(im*(k-1)*x)

# ╔═╡ 672fe091-3c8d-4281-97c7-70bb5fcc27b4
function make_k_indices(pselect, N)
	klist = Set{Int64}()
	while length(klist) < N
		push!(klist, rand(pselect))
	end
	return collect(klist)
end

# ╔═╡ 90d8f724-6fa2-409a-9780-1c797f51f605
#create synthetic regression data with 1 input and 1 output with a given complexity.  the complexity dictates how many fourier components there are
function make_synthetic_data(complexity, n, seed = 1234)
	pselect = 1:complexity*2
	Random.seed!(seed)
	klist = make_k_indices(pselect, complexity)
	#these constants ensure the variance of the output is about 1
	synthetic_y(x) = sum((2/sqrt(length(klist)))*real(e.(klist, x)))
	x = LinRange(π/2, π, n)
	y = synthetic_y.(x)
	#this shifting of the input axis is to ensure variance of the input is about 1
	(input = sqrt(3*π/2) .* (Float32.(x) .- 3*π/4), output = Float32.(y))
end

# ╔═╡ 90ca73ef-010d-408f-86b6-b8e9fe29e145
function plotdata(data, noisevar = 0.0; maxinds = 10000) 
	l = size(data.input, 1)
	inds = if l > maxinds
		shuffle(1:l)[1:maxinds]
	else
		1:l
	end

	x = data.input[inds]
	y = data.output[inds] .+ (sqrt(noisevar) .* randn(length(inds)))
	t = scatter(x = x, y = y, mode = "markers", marker_size = 1.0)
	plot(t)
end

# ╔═╡ 43a136a3-5260-49a9-9293-13ceb9b2d8b1
md"""
## NN Training
"""

# ╔═╡ 86b82142-438d-47c0-9d98-e48beba619b3
md"""
### LR Schedules
"""

# ╔═╡ 95a2d335-41e4-458d-a288-3396e679f9a0
function forminvschedule(αmax, numepochs)
	numepochs < 200 && error("Not enough epochs selected")
	ramp = LinRange(αmax / 100, αmax, 100)
	# ramp = fill(αmax, 100)
	decaysteps = numepochs - 200
	decayvec = Float32.(αmax ./ sqrt.(1:decaysteps))
	quench = LinRange(decayvec[end], decayvec[end] / 100, 101)
	return vcat(ramp, decayvec, quench)
end

# ╔═╡ 46c6e75e-aef9-466f-af31-06d978aba0ef
function formlinearschedule(αmax, numepochs; ramplength = 100)
	αmin = αmax * 1f-5
	numepochs < ramplength && return LinRange(αmin, αmax, numepochs) #only use linear ramp if fewer than ramplength epochs selected
	ramp = LinRange(αmin, αmax, ramplength)
	decaysteps = numepochs - ramplength
	decayvec = LinRange(αmax, αmin, decaysteps)
	vcat(ramp, decayvec)
end

# ╔═╡ 7b9e4677-4007-43f4-ac1b-a9532b14f89a
formconstantschedule(αmax, numepochs) = fill(αmax, numepochs)

# ╔═╡ 688e19d0-3e95-4b8b-bf3e-120836d2b7e4
defaultschedule(αmax, numepochs) = Vector{Float32}()

# ╔═╡ 75a60289-a983-48c2-a8c6-a9c3676735d1
namemap = Dict(
	formconstantschedule => "Constant",
	formlinearschedule => "Linear",
	forminvschedule => "Inverse Square Root",
	defaultschedule => "Default"
)

# ╔═╡ f4d73f34-fe94-4829-8c7b-ff0bc67cd05c
md"""
### Training Trials
"""

# ╔═╡ 600eb9ef-1aa0-4d15-9f86-0a4b8f3c03a9
function convertparams(rawparams)
	f = IOBuffer(rawparams)
	params = readBinParams(f)
	close(f)
	params[1][1], params[1][2]
end

# ╔═╡ 9a183762-648b-4713-b87e-c6d92c865a89
function get_train_key(data, layersize, nlayers, α; 
	formschedule = defaultschedule, 
	use_μP = false, 
	batchsize = 128, 
	numepochs = 5000, 
	λ = 0.0f0, 
	maxnorm = Inf,
	backend = getBackend(),
	costfunc = "sqErr",
	swa = false, 
	seed = 1234)

	(datahash = hash(data), layersize = layersize, nlayers = nlayers, α = α, lrschedule = namemap[formschedule], use_μP = use_μP, batchsize = batchsize, numepochs = numepochs, λ = λ, maxnorm = maxnorm, backend = backend, costfunc = costfunc, swa = swa, seed = seed)
end

# ╔═╡ 48c041d0-b804-46ce-a6b4-1f43a0ed0c58
"""
	check_learning_curve(data, layersize, nlayers, α; 
		formschedule = defaultschedule, 
		use_μP = false, 
		batchsize = 128, 
		numepochs = 5000, 
		λ = 0.0f0, 
		maxnorm = Inf,
		backend = getBackend(),
		costfunc = "sqErr",
		swa = false, 
		seed = 1234,
		resLayers = 0, 
		initvar = 1.0f0,
		overwrite = false))

Perform a training run on data which is a tuple containing input and output data where each row of data is an example.  Required arguments are the data, the width of each layer, the total number of layers, and the learning rate α.  Keyword arguments can control other aspects of training including batchsize and num epochs.  Default values are shown here.  If a training run with the same parameters is requested again then it will search for the results in a dictionary of cached outputs.
"""
function check_learning_curve(data, layersize, nlayers, α; 
	formschedule = defaultschedule, 
	use_μP = false, 
	batchsize = 128, 
	numepochs = 5000, 
	λ = 0.0f0, 
	maxnorm = Inf,
	backend = getBackend(),
	costfunc = "sqErr",
	swa = false, 
	seed = 1234,
	resLayers = 0, 
	initvar = 1.0f0,
	overwrite = false)

	lrschedule = formschedule(α, numepochs)

	trainkey = (datahash = hash(data), layersize = layersize, nlayers = nlayers, α = α, lrschedule = namemap[formschedule], use_μP = use_μP, batchsize = batchsize, numepochs = numepochs, λ = λ, maxnorm = maxnorm, backend = backend, costfunc = costfunc, swa = swa, seed = seed)

	if resLayers != 0
		trainkey = merge(trainkey, (resLayers = resLayers,))
	end

	if initvar != 1
		trainkey = merge(trainkey, (initvar = initvar,))
	end

	#convert data into Float32 matrices
	input = Float32.(data.input[:, :])
	output = Float32.(data.output[:, :])

	hidden = fill(layersize, nlayers)

	#look in currently loaded dictionary and json dictionary loaded from disk to see if a training run with these parameters has already been conducted
	if !overwrite && haskey(trainresultsdict, trainkey) 
		(costrecord, rawparams) = trainresultsdict[trainkey]
		(θ, β) = convertparams(rawparams)
	else
		trainfunc = backend == :GPU ? FCANN.ADAMAXTrainNNGPU : FCANN.ADAMAXTrainNNCPU
		Random.seed!(seed)
		θ_0, β_0 = FCANN.initializeparams_saxe(size(input, 2), hidden, size(output, 2), resLayers, use_μP = use_μP)

		if initvar != 1
			for i in eachindex(θ_0)
				θ_0[i] .*= sqrt(initvar)
				β_0[i] .*= sqrt(initvar)
			end
		end
	
		Random.seed!(seed)
		#add a key for training so results can be loaded if they already exist
		trainedout = trainfunc(
			[(input, output)], batchsize, θ_0, β_0, numepochs, size(input, 2), hidden, λ, maxnorm, alpha = α, 
			patience = 10, #stop training if no improvement for 100 epochs
			lrschedule = lrschedule, 
			minepoch = 0, 
			costFunc = "sqErr", use_μP = use_μP, swa = swa, resLayers = resLayers) 
	
		costrecord = trainedout[4]
		θ = trainedout[1]
		β = trainedout[2]

		
	end

	# if !overwrite && haskey(trainedoutputdict, trainkey)
	# 	println("Retrieving previously saved output from dictionary")
	# 	trainedoutput, error = trainedoutputdict[trainkey]
	# else
	# 	trainedoutput, error = FCANN.calcOutputCPU(input, output, θ, β, costFunc=costfunc)
	# 	# problem doing this concurrently
	# 	# trainedoutputdict[trainkey] = (output = trainedoutput, error = error)
	# end
	# (costrecord = costrecord, α = α, error = error, output = trainedoutput, nparams = FCANN.getNumParams(size(input, 2), hidden, size(output, 2)), trainkey = trainkey, θ = θ, β = β)
	(costrecord = costrecord, α = α, nparams = FCANN.getNumParams(size(input, 2), hidden, size(output, 2)), trainkey = trainkey, θ = θ, β = β)
	
end

# ╔═╡ 754a6fbd-2303-4ace-b6ec-da639402639e
#compare the output of a trained model with the data plotting the actual function curve
function check_output(args...; kwargs...)
	(data, layersize, nlayers, α) = args
	trainout = check_learning_curve(args...; kwargs...)
	data = args[1]
	x = Float32.(data.input[:, :])
	y = Float32.(data.output[:, :])
	modely, error = FCANN.calcOutputCPU(x, y, trainout.θ, trainout.β, costFunc=trainout.trainkey.costfunc)
	t1 = scatter(x = x[:], y = y[:], name = "data")
	t2 = scatter(x = x[:], y = modely[:], name = "model")
	title = """width: $(layersize) depth: $nlayers nparams: $(trainout.nparams) npoints: $(size(x, 1)) error: $(round(error, sigdigits = 3)) data var: $(round(var(y), sigdigits = 3))"""
	p = plot([t1, t2])
	md"""
	##### $title
	$p
	"""
end

# ╔═╡ 0c3f2430-c62c-41d8-aec9-512d8afccf91
"""
	check_learning_curve_trials(args...; nruns = 10, kwargs...)
Perform multiple training runs with the same parameters and different random seeds.  The seeds are integers increasing from 1 to nruns.  The return value is a named tuple containing a vector of cost records for each run, the final error value for each run, and summary statistics of the errors.
"""
function check_learning_curve_trials(args...; nruns = 5, kwargs...)
	(data, layersize, nlayers, α) = args
	
	#in this case use multiple threads
	if (haskey(kwargs, :backend) && kwargs[:backend] == :CPU) || (getBackend() == :CPU)
		output = Vector{Any}(undef, nruns)
		@threads for s in 1:nruns
		# for s in 1:nruns
			BLAS.set_num_threads(10)
			output[s] = check_learning_curve(args...; seed = s, kwargs...)
		end
	else
		output = [check_learning_curve(args...; seed = s, kwargs...) for s in 1:nruns]
	end

	println("Completed training runs and preparing to write results to dictionary and files")
	#add results to dictionary outside of thread loop
	for result in output
		# trainedoutputdict[result.trainkey] = (output = result.output, error = result.error)
		# println("Saved output in dictionary and preparing to save params")
		f = IOBuffer()
		FCANN.writeparams!(f, [(result.θ, result.β)])
		trainresultsdict[result.trainkey] = (costrecord = result.costrecord, params = copy(f.data))
		close(f)
	end
	
	f(field) = [getfield(a, field) for a in output]
	costrecords = f(:costrecord)
	errors = minimum.(costrecords)
	errorstats = summarystats(errors)
	(costrecords = costrecords, errors = errors, errorstats = errorstats)
end

# ╔═╡ ce857c92-485c-4632-8542-1a802604df2f
"""
	check_learning_curve_trials2(data; nruns = 5, layersize = 16, nlayers = 2, α = 0.02f0, kwargs...)
Perform multiple training runs with the same parameters and different random seeds.  The seeds are integers increasing from 1 to nruns.  The return value is a named tuple containing a vector of cost records for each run, the final error value for each run, and summary statistics of the errors.
"""
function check_learning_curve_trials2(data; nruns = 5, width = 16, depth = 2, α = 0.02f0, kwargs...)

	train(seed) = check_learning_curve(data, width, depth, α; seed = seed, kwargs...)
	
	#in this case use multiple threads
	if (haskey(kwargs, :backend) && kwargs[:backend] == :CPU) || (getBackend() == :CPU)
		output = Vector{Any}(undef, nruns)
		@threads for s in 1:nruns
		# for s in 1:nruns
			BLAS.set_num_threads(10)
			output[s] = train(s)
		end
	else
		output = [train(s) for s in 1:nruns]
	end

	println("Completed training runs and preparing to write results to dictionary and files")
	#add results to dictionary outside of thread loop
	for result in output
		# trainedoutputdict[result.trainkey] = (output = result.output, error = result.error)
		# println("Saved output in dictionary and preparing to save params")
		f = IOBuffer()
		FCANN.writeparams!(f, [(result.θ, result.β)])
		trainresultsdict[result.trainkey] = (costrecord = result.costrecord, params = copy(f.data))
		close(f)
	end
	
	f(field) = [getfield(a, field) for a in output]
	costrecords = f(:costrecord)
	errors = minimum.(costrecords)
	errorstats = summarystats(errors)
	(costrecords = costrecords, errors = errors, errorstats = errorstats, trainkey = (first(output).trainkey..., nruns = nruns))
end

# ╔═╡ 30cafdb0-8083-4dee-ac8e-fa9b06b12de1
md"""
## Train Results Plotting
"""

# ╔═╡ 2f5a793f-37d4-4482-91db-e46a7e562f24
"""
	plot_learning_curve_trials(args...; formschedule = formconstantschedule, nruns = 10, kwargs...)
Perform multiple training runs with the same parameters and different random seeds and plot the learning curve results with a separate line for each run.
"""
function plot_learning_curve_trials(args...; formschedule = formconstantschedule, kwargs...)
	(data, layersize, nlayers, α) = args
	results = check_learning_curve_trials(args...; formschedule = formschedule, kwargs...)
	traces = [scatter(y = costrecord, name = "seed = $i") for (i, costrecord) in enumerate(results.costrecords)]
	
	title = 
		md"""#### $(namemap[formschedule]) LR Training Runs on $layersize width $nlayers depth Network with α = $(round(α, sigdigits = 3)), Median Error = $(round(results.errorstats.median, sigdigits = 3))"""
	
	p = plot(traces, Layout(yaxis_type = "log", xaxis_title = "10 Epochs", yaxis_title = "Error", showlegend=false))
	
	md"""
	$title
	$p
	"""
end

# ╔═╡ 559afa35-7093-4058-ad9c-869e8477c375
"""
	plot_param_sweep(data, param::Symbol, list::AbstractVector, name; kwargs...)
Generate a scatter trace showing median error across multiple runs for a sweep over a given parameter.  The rest of the model and training parameters are held fixed.
"""
function plot_param_sweep(data, param::Symbol, list::AbstractVector, name; kwargs...)
	trials = [check_learning_curve_trials2(data; NamedTuple([param => p])..., kwargs...) for p in list]
	trialerrors = [a.errorstats.median for a in trials]
	trainkey = mapreduce(a -> Dict(zip(keys(a.trainkey), values(a.trainkey))), intersect, trials)
	x = eltype(list) <: Number ? log2.(list) : string.(list)
	(scatter(x = x, y = trialerrors, name = name), trainkey)
end

# ╔═╡ 2af6b9c9-cd25-4ce8-9888-413a08c65984
"""
```julia
	plot_scale_curves(data, scaleparam, scalelist; sweep_param = α, 
		sweep_list = 2.0f0 .^ (-5:0); kwargs...)
```

Plot sweep curves as described in the documentation for `plot_sweep_curve` with a separate trace for each scaleparam in scalelist.  All other aspects of training are held constant.
"""
function plot_scale_curves(data, scaleparam, scalelist; sweep_param = :α, 
		sweep_list = 2.0f0 .^ (-5:0), kwargs...)
	
	sweeps = [plot_param_sweep(data, sweep_param, sweep_list, "$scaleparam = $p"; NamedTuple([scaleparam => p])..., kwargs...) for p in scalelist]

	makedict(nt) = Dict(zip(keys(nt), values(nt)))
	
	traces = [first(a) for a in sweeps]
	trainkey = mapreduce(a -> a[2], intersect, sweeps)
	
	xtitle = eltype(sweep_list) <: Number ? "log2($sweep_param)" : string(sweep_param)
	p = plot(traces, Layout(xaxis_title = xtitle, yaxis_title = "error", legend = attr(orientation = "h"), height = 500, yaxis_type = "log"))
	
	# paramstr = use_μP ? "μP" : "Standard Parametrization"
	# res_str = haskey(kwargs, :resLayers) ? """, Res Layers = $(kwargs[:resLayers])""" : """"""
	# title = 
	# 	md"""#### $numepochs Epochs, $(namemap[formschedule]) LR Schedule, Depth = $depth, 
	# 	Batch Size = $batchsize, Using $paramstr $res_str"""

	traininfo = NamedTuple(sort(filter(a -> !in(a[1], [:datahash, :costfunc, :λ, :maxnorm, :swa, :backend, :seed]), trainkey), by = a -> a[1]))
	
	title = md"""#### $traininfo"""

	md"""
	$title
	$p
	"""
end

# ╔═╡ 60809dcb-caf1-49e2-8bb6-d6e4d8e78316
"""
	plot_error_curve(data, width, depth, αlist; makename, kwargs...)
Generate a scatter trace showing median error across nruns runs for each learning rate in αlist.  The rest of the model and training parameters are held fixed.
"""
function plot_error_curve(data, width, depth, αlist; name = "width = $width, depth = $depth", batchsize = 16, kwargs...)
	trialerrors = [begin
		trials = check_learning_curve_trials(data, width, depth, α; batchsize = batchsize, kwargs...)
		trials.errorstats.median
	end
	for α in αlist]
	scatter(x = log2.(αlist), y = trialerrors, name = name)
end

# ╔═╡ 9dcf96f9-e3b3-458a-abde-52de8708a633
"""
```julia
	plot_width_curves(data, depth, widthlist; 
		αlist = 2.0f0 .^ (-5:0), 
		numepochs = 1000, 
		formschedule = formconstantschedule, 
		batchsize = 16, 
		use_μP = true, 
		kwargs...)
```

Plot error curves as decribed in the documentation for `plot_error_curve` with a separate trace for each width in widthlist.  All other aspects of training are held constant including: number of epochs, learning rate schedule type, batch size, type of parameter initialization, cost function, and any regularization techniques.
"""
function plot_width_curves(data, depth, widthlist; αlist = 2.0f0 .^ (-5:0), numepochs = 1000, formschedule = formconstantschedule, batchsize = 16, use_μP = true, kwargs...)
	paramstr = use_μP ? "μP" : "Standard Parametrization"
	res_str = haskey(kwargs, :resLayers) ? """, Res Layers = $(kwargs[:resLayers])""" : """"""
	title = 
		md"""#### $numepochs Epochs, $(namemap[formschedule]) LR Schedule, Depth = $depth, 
		Batch Size = $batchsize, Using $paramstr $res_str"""
	
	traces = [plot_error_curve(data, width, depth, αlist; numepochs = numepochs, formschedule = formschedule, batchsize = batchsize, use_μP = use_μP, name = "width = $width", kwargs...) for width in widthlist]
	
	p = plot(traces, Layout(xaxis_title = "log2(α)", yaxis_title = "error", legend = attr(orientation = "h"), height = 500, yaxis_type = "log"))

	md"""
	$title
	$p
	"""
end

# ╔═╡ dcb2575e-e111-4830-a9fc-614336440c4f
"""
```julia
	plot_depth_curves(data, depthlist, width; 
		αlist = 2.0f0 .^ (-5:0), 
		numepochs = 1000, 
		formschedule = formconstantschedule, 
		batchsize = 16, 
		use_μP = true, 
		kwargs...)
```

Plot error curves as decribed in the documentation for `plot_error_curve` with a separate trace for each depth in depthlist.  All other aspects of training are held constant including: number of epochs, learning rate schedule type, batch size, type of parameter initialization, cost function, and any regularization techniques.
"""
function plot_depth_curves(data, depthlist, width; αlist = 2.0f0 .^ (-5:0), numepochs = 1000, formschedule = formconstantschedule, batchsize = 16, use_μP = true, kwargs...)
	paramstr = use_μP ? "μP" : "Standard Parametrization"
	title = md"""#### $numepochs Epochs, $(namemap[formschedule]) LR Schedule, 
	Width = $width, Batch Size = $batchsize, Using $paramstr"""
	
	traces = [plot_error_curve(data, width, depth, αlist; numepochs = numepochs, formschedule = formschedule, batchsize = batchsize, use_μP = use_μP, name = "depth = $depth", kwargs...) for depth in depthlist]
	
	p = plot(traces, Layout(xaxis_title = "log2(α)", yaxis_title = "error", yaxis_type = "log", legend = attr(orientation = "h"), height = 500))

	md"""
	$title
	$p
	"""
end

# ╔═╡ de920afb-6a87-4056-937d-85b2f1b0cbcc
"""
```julia
	plot_batch_curves(data, depth, width, batchlist; 
		αlist = 2.0f0 .^ (-5:0), 
		numepochs = 1000, 
		formschedule = formconstantschedule, 
		use_μP = true, 
		kwargs...)
```

Plot error curves as decribed in the documentation for `plot_error_curve` with a separate trace for each depth in depthlist.  All other aspects of training are held constant including: number of epochs, learning rate schedule type, batch size, type of parameter initialization, cost function, and any regularization techniques.
"""
function plot_batch_curves(data, depth, width, batchlist; αlist = 2.0f0 .^ (-5:0), numepochs = 1000, formschedule = formconstantschedule, batchsize = 16, use_μP = true, kwargs...)
	paramstr = use_μP ? "μP" : "Standard Parametrization"
	title = md"""#### $numepochs Epochs, $(namemap[formschedule]) LR Schedule, Width = $width, 
	Depth = $depth, Using $paramstr"""
	
	traces = [plot_error_curve(data, width, depth, αlist; numepochs = numepochs, formschedule = formschedule, batchsize = batchsize, use_μP = use_μP, name = "batchsize = $batchsize", kwargs...) for batchsize in batchlist]

	p = plot(traces, Layout(xaxis_title = "log2(α)", yaxis_title = "error", yaxis_type = "log", legend = attr(orientation = "h"), height = 500))

	md"""
	$title
	$p
	"""
end

# ╔═╡ a2599ead-232a-419c-8fb0-08f43da0d3fd
"""
```julia
	plot_var_curves(data, depth, width, varlist; 
		αlist = 2.0f0 .^ (-5:0), 
		batchsize = 16,
		numepochs = 1000, 
		formschedule = formconstantschedule, 
		use_μP = true, 
		kwargs...)
```

Plot error curves as decribed in the documentation for `plot_error_curve` with a separate trace for each depth in depthlist.  All other aspects of training are held constant including: number of epochs, learning rate schedule type, batch size, type of parameter initialization, cost function, and any regularization techniques.
"""
function plot_var_curves(data, depth, width, varlist; αlist = 2.0f0 .^ (-5:0), numepochs = 1000, formschedule = formconstantschedule, batchsize = 16, use_μP = true, kwargs...)
	makename(var) = "init variance = $var"
	paramstr = use_μP ? "μP" : "Standard Parametrization"
	title = md"""#### $numepochs Epochs, $(namemap[formschedule]) LR Schedule, Width = $width, 
	Depth = $depth, Using $paramstr"""
	
	traces = [plot_error_curve(data, width, depth, αlist; initvar = initvar, numepochs = numepochs, formschedule = formschedule, batchsize = batchsize, use_μP = use_μP, name = "init variance = $initvar", kwargs...) for initvar in varlist]

	p = plot(traces, Layout(xaxis_title = "log2(α)", yaxis_title = "error", yaxis_type = "log", legend = attr(orientation = "h"), height = 500))

	md"""
	$title
	$p
	"""
end

# ╔═╡ c241f9a0-3a26-4d17-9346-63ba0ac3aeea
md"""
# Visualize Results
"""

# ╔═╡ b259b4f8-7144-4550-93b2-7eaac1d8f111
md"""
## Save Results
"""

# ╔═╡ 9c1eaaf7-f911-4504-824f-ce3a169ba60d
@bind savedict Button("Save train results")

# ╔═╡ f3997572-ec5d-4544-bce6-7c8405cc86fa
md"""
## Data Sets
"""

# ╔═╡ dc9897d7-5a16-42f7-8294-d9372b9be8af
smalldata = make_synthetic_data(40, 2^10, 1) #make data with a seed of 1)

# ╔═╡ ac14ce0f-2247-4e28-8c5b-2d925bea8a79
largedata = make_synthetic_data(500, 2^20, 1) #make data with a seed of 1)

# ╔═╡ 72354352-08ec-4564-bd7d-8b885da8e03f
plotdata(smalldata, 0.00)

# ╔═╡ 1cbd7214-0ec3-4ded-b544-426d7ff8ef53
plotdata(largedata, 0.00)

# ╔═╡ aa78e3af-85b9-4970-b8af-9b0147d460ce
md"""
## Learning Curves
"""

# ╔═╡ c89901b5-2358-4f88-ad5b-8ef729b792e4
md"""
## Error Curves
"""

# ╔═╡ ba989756-ba40-49c2-8a15-c822d437bdad
md"""
### Width Curves
"""

# ╔═╡ 182ad9e5-4281-42a1-ad06-6cd0a384f29c
@bind width_sweep_params confirm(PlutoUI.combine() do Child
	md"""
	Depth: $(Child("depth", NumberField(1:16, default = 2)))
	Res Layers: $(Child("resLayers", NumberField(0:16, default = 0)))
	Batch Size: $(Child("batchsize", NumberField(4:256, default = 32)))
	Num Epochs: $(Child("epochs", NumberField(10:5000, default = 1000)))
	LR Schedule: $(Child("formschedule", Select([a[1] => a[2] for a in namemap], default = formconstantschedule)))
	Training Runs: $(Child("nruns", NumberField(1:100, default = 15)))
	Use μP: $(Child("use_μP", CheckBox()))
	Log2 Layer Width: Min $(Child("log2widthmin", NumberField(0:256, default = 1))) Max $(Child("log2widthmax", NumberField(0:256, default = 5)))
	Log2 $(Child("sweep_param", Select([:α, :initvar]))): Min $(Child("log2sweepmin", NumberField(-20:.1:10, default = -2))) Max $(Child("log2sweepmax", NumberField(-20:.1:10, default = 2)))
	"""
end)

# ╔═╡ 40e68fde-7451-4a7f-986e-0a900158ce7c
begin
width_sweep_other_param = first(setdiff((:α, :initvar), (width_sweep_params.sweep_param,)))
width_sweep_other_default = (width_sweep_other_param == :α) ? -2 : 0
md"""
Log2 $width_sweep_other_param $(@bind width_sweep_other NumberField(-20:.1:20, default = width_sweep_other_default))
"""
end

# ╔═╡ dd950d58-8644-4ad8-9bdf-e0c880895158
# ╠═╡ show_logs = false
plot_scale_curves(smalldata, :width, 2 .^ (width_sweep_params.log2widthmin:width_sweep_params.log2widthmax); sweep_param = width_sweep_params.sweep_param, sweep_list = 2.0f0 .^ (width_sweep_params.log2sweepmin:width_sweep_params.log2sweepmax), batchsize = width_sweep_params.batchsize, formschedule = width_sweep_params.formschedule, nruns = width_sweep_params.nruns, backend = :CPU, numepochs = width_sweep_params.epochs, use_μP = width_sweep_params.use_μP, NamedTuple([width_sweep_other_param => Float32(2 ^ width_sweep_other)])...)

# ╔═╡ 797bbfb3-87c1-47b0-821b-b35c62705ed4
md"""
### Depth Curves
"""

# ╔═╡ 289781e3-8bbe-40f2-b362-66549d143530
@bind depth_sweep_params confirm(PlutoUI.combine() do Child
	md"""
	Log2 Width: $(Child("log2width", NumberField(1:16, default = 4)))
	Res Layers: $(Child("resLayers", NumberField(0:16, default = 0)))
	Batch Size: $(Child("batchsize", NumberField(4:256, default = 32)))
	Num Epochs: $(Child("epochs", NumberField(10:5000, default = 1000)))
	LR Schedule: $(Child("formschedule", Select([a[1] => a[2] for a in namemap], default = formconstantschedule)))
	Training Runs: $(Child("nruns", NumberField(1:100, default = 15)))
	Use μP: $(Child("use_μP", CheckBox()))
	Depth: Min $(Child("mindepth", NumberField(0:256, default = 1))) Max $(Child("maxdepth", NumberField(0:256, default = 4)))
	Log2 $(Child("sweep_param", Select([:α, :initvar]))): Min $(Child("log2sweepmin", NumberField(-20:.1:10, default = -2))) Max $(Child("log2sweepmax", NumberField(-20:.1:10, default = 2)))
	"""
end)

# ╔═╡ d2e3562a-56e3-4c2c-ac71-1f757910a777
begin
depth_sweep_other_param = first(setdiff((:α, :initvar), (depth_sweep_params.sweep_param,)))
depth_sweep_other_default = (depth_sweep_other_param == :α) ? -2 : 0
md"""
Log2 $depth_sweep_other_param $(@bind depth_sweep_other NumberField(-20:.1:20, default = depth_sweep_other_default))
"""
end

# ╔═╡ e7f6b5c2-19d5-403b-908b-147ef4190697
# ╠═╡ show_logs = false
plot_scale_curves(smalldata, :depth, depth_sweep_params.mindepth:depth_sweep_params.maxdepth; width = 2 ^ depth_sweep_params.log2width, sweep_param = depth_sweep_params.sweep_param, sweep_list = 2.0f0 .^ (depth_sweep_params.log2sweepmin:depth_sweep_params.log2sweepmax), batchsize = depth_sweep_params.batchsize, formschedule = depth_sweep_params.formschedule, nruns = depth_sweep_params.nruns, backend = :CPU, numepochs = depth_sweep_params.epochs, use_μP = depth_sweep_params.use_μP, NamedTuple([depth_sweep_other_param => Float32(2 ^ depth_sweep_other)])...)

# ╔═╡ acb65a8f-2342-433f-a614-c00087e03eae
# ╠═╡ show_logs = false
plot_scale_curves(smalldata, :numepochs, [10, 100, 200, 400, 800], backend = :CPU, use_μP = true, sweep_list = 2.0f0 .^ (-3:2), formschedule = formconstantschedule, nruns = 15, batchsize = 64)

# ╔═╡ 52cf3929-8d11-4bff-b44d-e17f18403187
# ╠═╡ show_logs = false
plot_scale_curves(smalldata, :width, 2 .^ (1:6), sweep_param = :formschedule, sweep_list = [formconstantschedule, forminvschedule, formlinearschedule, defaultschedule], backend = :CPU, use_μP = true, nruns = 15, batchsize = 64, numepochs = 1000, α = 0.5f0)

# ╔═╡ 134d132e-c600-4dc9-b25e-133505f78425
# ╠═╡ show_logs = false
plot_scale_curves(smalldata, :depth, 1:6, sweep_param = :formschedule, sweep_list = [formconstantschedule, forminvschedule, formlinearschedule, defaultschedule], backend = :CPU, use_μP = true, nruns = 15, batchsize = 64, numepochs = 1000, α = 0.25f0)

# ╔═╡ 98a204c3-0b84-4acd-9113-fca3afaf19da
# ╠═╡ show_logs = false
plot_scale_curves(smalldata, :batchsize, [8, 16, 32, 64, 128, 256], backend = :CPU, use_μP = true, sweep_list = 2.0f0 .^ (-3:2), nruns = 15, numepochs = 1000, formschedule = formconstantschedule)

# ╔═╡ dd112814-1f90-468c-a395-ed41fc1b0db2
# ╠═╡ show_logs = false
plot_scale_curves(largedata, :width, [32, 64, 128, 256, 512, 1024, 2048, 4096], backend = :GPU, use_μP = true, nruns = 15, numepochs = 10, sweep_list = 2.0f0 .^ (-6:3), formschedule = formconstantschedule, batchsize = 1024)

# ╔═╡ ea4870f3-8cbf-4140-923d-1de6161e958e
# ╠═╡ show_logs = false
plot_scale_curves(largedata, :batchsize, [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], backend = :CPU, use_μP = true, nruns = 15, numepochs = 1, sweep_list = 2.0f0 .^ (-6:3), formschedule = formconstantschedule)

# ╔═╡ 1dcf974b-4783-457c-b096-f7308f62ade0
plot_scale_curves(largedata, :batchsize, [16, 32, 64, 128, 256, 512, 1024, 2048, 4096], backend = :CPU, width = 64, use_μP = true, nruns = 15, numepochs = 1, sweep_list = 2.0f0 .^ (-6:3), formschedule = formconstantschedule)

# ╔═╡ f0a46bee-78e7-4f25-bfb1-97ee1de6fc5a
# ╠═╡ show_logs = false
plot_width_curves(smalldata, 2, [16, 32, 64, 128, 256], batchsize = 32, nruns = 15, αlist = 2.0f0 .^ (-4:2), backend = :CPU)

# ╔═╡ 5fc80d33-2046-454c-b681-db964683056d
# ╠═╡ show_logs = false
plot_var_curves(smalldata, 2, 16, 2.0f0 .^ (-4:4), nruns = 15, backend = :CPU, αlist = 2.0f0 .^ (-4:2))

# ╔═╡ 2e262170-71bb-4ef1-a899-3e1e0f32fa4c
# ╠═╡ show_logs = false
plot_width_curves(smalldata, 2, [16, 32, 64, 128, 256], batchsize = 32, nruns = 15, αlist = 2.0f0 .^ (-10:3), backend = :CPU, use_μP = false)

# ╔═╡ ff110d2e-1f8e-4931-9fae-ab8a2e855da0
# ╠═╡ show_logs = false
plot_width_curves(smalldata, 2, [2, 4, 8, 16, 32, 64], batchsize = 128, nruns = 15, use_μP = false, αlist = 2f0 .^ (-10:3), backend = :CPU)

# ╔═╡ 89aa0c7f-4d17-47c7-8cf9-6df73d34dd73
# ╠═╡ show_logs = false
plot_width_curves(smalldata, 2, [2, 4, 8, 16, 32, 64], nruns = 15, formschedule = formlinearschedule, backend = :CPU)

# ╔═╡ 6923cb80-4dbc-4f5c-bacb-414c3425817b
# ╠═╡ show_logs = false
plot_width_curves(smalldata, 2, [2, 4, 8], nruns = 15, backend = :CPU, αlist = 2.0f0 .^ (-1:3), resLayers = 1)

# ╔═╡ 2436d324-f586-416d-b6d8-f94558b231cb
# ╠═╡ show_logs = false
plot_var_curves(smalldata, 2, 8, 2.0f0 .^ (-6:4), nruns = 15, backend = :CPU, resLayers = 1, αlist = 2.0f0 .^ (-1:1))

# ╔═╡ 15ef09a5-518b-45f9-91e1-2d700145417e
# ╠═╡ show_logs = false
plot_depth_curves(smalldata, [1, 2, 4, 8, 16], 32, αlist = 2f0 .^ (-7:1), backend = :CPU, nruns = 15, batchsize = 64)

# ╔═╡ 320a00a8-f15e-4174-9e64-8ad73cb9660b
md"""
### Batch Curves
"""

# ╔═╡ 2f3afb5a-5402-4499-a5b3-291a6b58c9a9
# ╠═╡ show_logs = false
plot_batch_curves(smalldata, 4, 16, [8, 16, 32, 64, 128], nruns = 15, backend = :CPU)

# ╔═╡ 7944326d-ce51-42c1-ab8a-3c6ac9b14da3
# ╠═╡ show_logs = false
plot_batch_curves(smalldata, 4, 32, [8, 16, 32, 64, 128], nruns = 15, backend = :CPU)

# ╔═╡ 1b84abfd-3fe3-4c58-84bd-bf2255b6f889
# ╠═╡ show_logs = false
plot_batch_curves(smalldata, 4, 64, [8, 16, 32, 64, 128], nruns = 15, backend = :CPU)

# ╔═╡ d4cc6589-3bb8-455b-8ed4-7536af7d6ed6
# ╠═╡ show_logs = false
plot_batch_curves(smalldata, 4, 128, [8, 16, 32, 64, 128], nruns = 15, backend = :CPU)

# ╔═╡ f5f46495-3c39-4c95-8458-af790b4f8c9e
md"""
### Width Curves
"""

# ╔═╡ 9430417f-0da4-491d-83af-aaece0a6f92e
# ╠═╡ show_logs = false
plot_width_curves(smalldata, 2, [2, 4, 8, 16, 32, 64], nruns = 15, use_μP = false, backend=:CPU)

# ╔═╡ c58165e0-9263-453a-bbfc-b346e71561e9
# ╠═╡ show_logs = false
plot_width_curves(smalldata, 2, [2, 4, 8, 16, 32, 64], nruns = 15, batchsize = 64, backend=:CPU)

# ╔═╡ 58b3e9c0-bf72-4f18-83de-e63a0fbb069a
# ╠═╡ show_logs = false
plot_width_curves(smalldata, 2, [2, 4, 8, 16, 32, 64, 128], nruns = 15, batchsize = 64, use_μP = false, αlist = 2f0 .^ (-7:-2), backend = :CPU)

# ╔═╡ 42427347-3dad-4c6e-a24c-1ec7475bc89b
# ╠═╡ show_logs = false
plot_width_curves(smalldata, 4, [2, 4, 8, 16, 32, 64, 128], nruns = 15, batchsize = 64, backend=:CPU)

# ╔═╡ d86d0205-a794-4b7e-88d9-7f674aa8c254
# ╠═╡ show_logs = false
plot_width_curves(smalldata, 4, [2, 4, 8, 16, 32, 64, 128], nruns = 15, batchsize = 64, backend=:CPU, use_μP = false, αlist = 2f0 .^ (-10:-4))

# ╔═╡ 81a6baa4-2025-4e6a-9034-76bf925a9e97
# ╠═╡ show_logs = false
check_output(smalldata, 32, 4, 0.25f0, numepochs = 2000, formschedule = formlinearschedule, batchsize = 32, use_μP = true, backend = :CPU, costfunc = "sqErr")

# ╔═╡ af70c9ae-fa72-47aa-bf7c-06cd6e530bc0
# ╠═╡ show_logs = false
check_output(smalldata, 32, 2, 0.25f0, numepochs = 1000, formschedule = formlinearschedule, batchsize = 32, use_μP = false, backend = :CPU)

# ╔═╡ bcbbbbfe-523f-4fa4-b9a7-0f6f15826ed2
function makemultset(numexamples)
	Random.seed!(1)
	xs = 3.14f0 .* (rand(Float32, numexamples) .- 0.5f0)
	ys = 3.14f0 .* (rand(Float32, numexamples) .- 0.5f0)
	out = xs .* ys .* 1.25f0
	(input = [xs ys], output = out[:, [1]])
end

# ╔═╡ c02688ea-a4c5-4d58-9678-b3cf542d18e0
multset = makemultset(1024)

# ╔═╡ 242e1864-5db8-42f9-abfb-e3c772fa55b4
function makemaxset(numexamples)
	Random.seed!(1)
	xs = 3.14f0 .* (rand(Float32, numexamples) .- 0.5f0)
	ys = 3.14f0 .* (rand(Float32, numexamples) .- 0.5f0)
	out = [max(xs[i], ys[i]) for i in eachindex(xs)]
	(input = [xs ys], output = out[:, [1]])
end

# ╔═╡ 4d0e6b28-11fd-4bec-a7c4-279622e0a77f
maxset = makemaxset(1024)

# ╔═╡ d68bb770-b852-4e20-b3bd-c86128b1e772
function makesortset(numexamples, d)
	Random.seed!(1)
	inputs = [3.14f0 .* (rand(Float32, numexamples) .- 0.5f0) for _ in 1:d]
	out = mapreduce(a -> a', vcat, [sort([input[i] for input in inputs]) for i in eachindex(inputs[1])])
	(input = reduce(hcat, inputs), output = out)
end

# ╔═╡ d812b8b8-b7ab-4bc2-988a-a0af11b68083
sortset = makesortset(2^20, 3)

# ╔═╡ aa9e8ada-3270-46c2-92e8-11ad97946930
dataset_dict = Dict(hash(a) => a for a in [smalldata, largedata, maxset, sortset, multset])

# ╔═╡ 6fbd314c-376c-42c7-bd8a-a52bd7fb112f
datasets = [smalldata => "smalldata", maxset => "maxset", sortset => "sortset", multset => "multset", largedata => "largedata"]

# ╔═╡ 396acc7d-db71-4c48-b432-66f1ce29f58c
datasetnamedict = Dict(datasets)

# ╔═╡ 685535e1-e59b-4582-8029-03822aec228d
@bind trainparams confirm(PlutoUI.combine() do Child
	md"""
	Data Set: $(Child("dataset", Select(datasets)))
	Layer Width: $(Child("width", NumberField(1:256, default = 32)))
	Depth: $(Child("depth", NumberField(1:16, default = 2)))
	Res Layers: $(Child("resLayers", NumberField(0:16, default = 0)))
	Log2 Learning Rate: $(Child("log2α", NumberField(-20:.1:10, default = -2)))
	Batch Size: $(Child("batchsize", NumberField(4:256, default = 32)))
	Training Runs: $(Child("nruns", NumberField(1:100, default = 15)))
	Num Epochs: $(Child("epochs", NumberField(10:5000, default = 1000)))
	LR Schedule: $(Child("formschedule", Select([a[1] => a[2] for a in namemap], default = formconstantschedule)))
	Use μP: $(Child("use_μP", CheckBox()))
	Log2 Init Variance: $(Child("initvar", NumberField(-4.:.1:4., default = 0)))
	"""
end)

# ╔═╡ 8031d125-7940-4cb4-b248-560287ba30d5
# ╠═╡ show_logs = false
plot_learning_curve_trials(smalldata, trainparams.width, trainparams.depth, Float32(2. ^ trainparams.log2α), backend = :CPU, use_μP = trainparams.use_μP, batchsize = trainparams.batchsize, numepochs = trainparams.epochs, nruns = trainparams.nruns, formschedule = trainparams.formschedule, resLayers = trainparams.resLayers, initvar = Float32(2. ^ trainparams.initvar))

# ╔═╡ e6466a82-0534-436f-b4e4-0e173303777f
# ╠═╡ show_logs = false
plot_scale_curves(sortset, :batchsize, [8, 16, 32, 64, 128, 256], backend = :CPU, use_μP = true, sweep_list = 2.0f0 .^ (-5:0), nruns = 15, numepochs = 1, formschedule = formconstantschedule)

# ╔═╡ 941f2b8a-a51f-4b2a-b2a0-64a3ebf2df55
# ╠═╡ show_logs = false
plot_scale_curves(sortset, :batchsize, [8, 16, 32, 64, 128, 256], backend = :CPU, use_μP = true, sweep_list = 2.0f0 .^ (-5:0), nruns = 15, numepochs = 1, formschedule = formconstantschedule, width = 4)

# ╔═╡ f4e40ef7-c8e1-433e-8fc3-936f01146428
# ╠═╡ show_logs = false
plot_width_curves(multset, 2, [1, 2, 3, 4], nruns = 5, formschedule = formconstantschedule, batchsize = 32, αlist = 2f0 .^ (-7:-4), backend = :CPU)

# ╔═╡ b8a416ac-d910-4f9d-930c-74ed69b18a8c
# ╠═╡ show_logs = false
plot_width_curves(maxset, 2, [1, 2, 3, 4], nruns = 5, formschedule = formconstantschedule, batchsize = 32, αlist = 2f0 .^ (-7:-4), backend = :CPU)

# ╔═╡ 4526797b-0b54-4b65-81eb-3d3befbfb3d7
# ╠═╡ show_logs = false
plot_width_curves(sortset, 4, [8, 16, 32, 64, 128], nruns = 15, formschedule = formconstantschedule, batchsize = 8192, αlist = 2f0 .^ (-3:1), numepochs = 10)

# ╔═╡ 663e8690-87de-4b05-9355-fe10d373728e
# ╠═╡ show_logs = false
plot_width_curves(sortset, 4, [8, 16, 32, 64, 128], nruns = 15, formschedule = formconstantschedule, batchsize = 32, αlist = 2f0 .^ (-6:-2), numepochs = 2, backend = :CPU)

# ╔═╡ b8ec16dd-c557-40bf-88e4-bb55f19eff9e
# ╠═╡ show_logs = false
plot_depth_curves(sortset, [1, 2, 4, 8, 16, 32], 16, nruns = 15, formschedule = formconstantschedule, batchsize = 32, αlist = 2f0 .^ (-8:0), numepochs = 2, backend = :CPU)

# ╔═╡ ff9d8e91-c9c7-4257-b4f6-ef62bbf89cb8
# ╠═╡ show_logs = false
plot_batch_curves(sortset, 4, 128, [32, 64, 128, 256, 512, 1024, 2048], nruns = 10, formschedule = formconstantschedule, αlist = 2f0 .^ (-6:0), numepochs = 2)

# ╔═╡ c7489681-213f-48db-94f1-f0b670f10350
# ╠═╡ show_logs = false
plot_batch_curves(sortset, 4, 32, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], nruns = 15, formschedule = formconstantschedule, αlist = 2f0 .^ (-6:0), numepochs = 2, backend = :CPU)

# ╔═╡ fcdcae67-5cbd-432b-8878-ab7ca7cd728a
# ╠═╡ show_logs = false
plot_batch_curves(sortset, 2, 32, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], nruns = 15, formschedule = formconstantschedule, αlist = 2f0 .^ (-6:0), numepochs = 2, backend = :CPU)

# ╔═╡ 0b37c368-ac33-451c-a75c-bf333ce33e60
# ╠═╡ show_logs = false
plot_batch_curves(sortset, 2, 32, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], nruns = 15, formschedule = formconstantschedule, αlist = 2f0 .^ (-10:-2), numepochs = 2, backend = :CPU, use_μP = false)

# ╔═╡ bbae926c-1b3e-4261-86f0-68be430cdec5
md"""
# Save and Load Results
"""

# ╔═╡ d734a6d6-7284-4844-909c-23a3733b3bc7
function savedata()
	jsonstr = JSON3.write(trainresultsdict)
	write("trainresults.json", jsonstr)
end

# ╔═╡ a23245ac-01fb-4b6b-8858-77e6d8483e6e
begin
	savedict
	savedata()
end

# ╔═╡ 115e3ddc-5b7a-4f66-b88a-3ce61e7005a1
global trainedoutputdict = Dict{NamedTuple, NamedTuple}()

# ╔═╡ d5390279-fbe1-42d8-9cfd-84f7e74fd34d
md"""
# Package Installation and Setup
"""

# ╔═╡ Cell order:
# ╟─33300f8c-ba16-11ed-08bd-6b6cdd6a420d
# ╟─1b8394bd-5dd4-4229-ac1f-507961f4b527
# ╟─f9f406cd-6082-4bac-8f34-53b7527fd002
# ╠═076086e4-12b4-4a7d-bbca-3bd54072ef07
# ╠═672fe091-3c8d-4281-97c7-70bb5fcc27b4
# ╠═90d8f724-6fa2-409a-9780-1c797f51f605
# ╠═90ca73ef-010d-408f-86b6-b8e9fe29e145
# ╟─43a136a3-5260-49a9-9293-13ceb9b2d8b1
# ╟─86b82142-438d-47c0-9d98-e48beba619b3
# ╠═95a2d335-41e4-458d-a288-3396e679f9a0
# ╠═46c6e75e-aef9-466f-af31-06d978aba0ef
# ╠═7b9e4677-4007-43f4-ac1b-a9532b14f89a
# ╠═688e19d0-3e95-4b8b-bf3e-120836d2b7e4
# ╠═75a60289-a983-48c2-a8c6-a9c3676735d1
# ╟─f4d73f34-fe94-4829-8c7b-ff0bc67cd05c
# ╠═600eb9ef-1aa0-4d15-9f86-0a4b8f3c03a9
# ╠═9a183762-648b-4713-b87e-c6d92c865a89
# ╠═48c041d0-b804-46ce-a6b4-1f43a0ed0c58
# ╠═754a6fbd-2303-4ace-b6ec-da639402639e
# ╠═0c3f2430-c62c-41d8-aec9-512d8afccf91
# ╠═ce857c92-485c-4632-8542-1a802604df2f
# ╟─30cafdb0-8083-4dee-ac8e-fa9b06b12de1
# ╠═2f5a793f-37d4-4482-91db-e46a7e562f24
# ╠═559afa35-7093-4058-ad9c-869e8477c375
# ╠═2af6b9c9-cd25-4ce8-9888-413a08c65984
# ╠═60809dcb-caf1-49e2-8bb6-d6e4d8e78316
# ╠═9dcf96f9-e3b3-458a-abde-52de8708a633
# ╠═dcb2575e-e111-4830-a9fc-614336440c4f
# ╠═de920afb-6a87-4056-937d-85b2f1b0cbcc
# ╠═a2599ead-232a-419c-8fb0-08f43da0d3fd
# ╟─c241f9a0-3a26-4d17-9346-63ba0ac3aeea
# ╟─b259b4f8-7144-4550-93b2-7eaac1d8f111
# ╟─9c1eaaf7-f911-4504-824f-ce3a169ba60d
# ╟─f3997572-ec5d-4544-bce6-7c8405cc86fa
# ╠═dc9897d7-5a16-42f7-8294-d9372b9be8af
# ╠═ac14ce0f-2247-4e28-8c5b-2d925bea8a79
# ╟─72354352-08ec-4564-bd7d-8b885da8e03f
# ╟─1cbd7214-0ec3-4ded-b544-426d7ff8ef53
# ╠═c02688ea-a4c5-4d58-9678-b3cf542d18e0
# ╠═4d0e6b28-11fd-4bec-a7c4-279622e0a77f
# ╠═d812b8b8-b7ab-4bc2-988a-a0af11b68083
# ╠═aa9e8ada-3270-46c2-92e8-11ad97946930
# ╠═6fbd314c-376c-42c7-bd8a-a52bd7fb112f
# ╠═396acc7d-db71-4c48-b432-66f1ce29f58c
# ╟─aa78e3af-85b9-4970-b8af-9b0147d460ce
# ╟─685535e1-e59b-4582-8029-03822aec228d
# ╟─8031d125-7940-4cb4-b248-560287ba30d5
# ╟─c89901b5-2358-4f88-ad5b-8ef729b792e4
# ╟─ba989756-ba40-49c2-8a15-c822d437bdad
# ╟─182ad9e5-4281-42a1-ad06-6cd0a384f29c
# ╟─40e68fde-7451-4a7f-986e-0a900158ce7c
# ╟─dd950d58-8644-4ad8-9bdf-e0c880895158
# ╟─797bbfb3-87c1-47b0-821b-b35c62705ed4
# ╟─289781e3-8bbe-40f2-b362-66549d143530
# ╟─d2e3562a-56e3-4c2c-ac71-1f757910a777
# ╟─e7f6b5c2-19d5-403b-908b-147ef4190697
# ╠═acb65a8f-2342-433f-a614-c00087e03eae
# ╠═52cf3929-8d11-4bff-b44d-e17f18403187
# ╠═134d132e-c600-4dc9-b25e-133505f78425
# ╠═98a204c3-0b84-4acd-9113-fca3afaf19da
# ╠═e6466a82-0534-436f-b4e4-0e173303777f
# ╠═941f2b8a-a51f-4b2a-b2a0-64a3ebf2df55
# ╠═dd112814-1f90-468c-a395-ed41fc1b0db2
# ╠═ea4870f3-8cbf-4140-923d-1de6161e958e
# ╠═1dcf974b-4783-457c-b096-f7308f62ade0
# ╠═f0a46bee-78e7-4f25-bfb1-97ee1de6fc5a
# ╠═5fc80d33-2046-454c-b681-db964683056d
# ╠═2e262170-71bb-4ef1-a899-3e1e0f32fa4c
# ╠═ff110d2e-1f8e-4931-9fae-ab8a2e855da0
# ╠═89aa0c7f-4d17-47c7-8cf9-6df73d34dd73
# ╠═6923cb80-4dbc-4f5c-bacb-414c3425817b
# ╠═2436d324-f586-416d-b6d8-f94558b231cb
# ╠═15ef09a5-518b-45f9-91e1-2d700145417e
# ╟─320a00a8-f15e-4174-9e64-8ad73cb9660b
# ╠═2f3afb5a-5402-4499-a5b3-291a6b58c9a9
# ╠═7944326d-ce51-42c1-ab8a-3c6ac9b14da3
# ╠═1b84abfd-3fe3-4c58-84bd-bf2255b6f889
# ╠═d4cc6589-3bb8-455b-8ed4-7536af7d6ed6
# ╟─f5f46495-3c39-4c95-8458-af790b4f8c9e
# ╠═9430417f-0da4-491d-83af-aaece0a6f92e
# ╠═c58165e0-9263-453a-bbfc-b346e71561e9
# ╠═58b3e9c0-bf72-4f18-83de-e63a0fbb069a
# ╠═42427347-3dad-4c6e-a24c-1ec7475bc89b
# ╠═d86d0205-a794-4b7e-88d9-7f674aa8c254
# ╠═81a6baa4-2025-4e6a-9034-76bf925a9e97
# ╠═af70c9ae-fa72-47aa-bf7c-06cd6e530bc0
# ╠═bcbbbbfe-523f-4fa4-b9a7-0f6f15826ed2
# ╠═242e1864-5db8-42f9-abfb-e3c772fa55b4
# ╠═d68bb770-b852-4e20-b3bd-c86128b1e772
# ╠═f4e40ef7-c8e1-433e-8fc3-936f01146428
# ╠═b8a416ac-d910-4f9d-930c-74ed69b18a8c
# ╠═4526797b-0b54-4b65-81eb-3d3befbfb3d7
# ╠═663e8690-87de-4b05-9355-fe10d373728e
# ╠═b8ec16dd-c557-40bf-88e4-bb55f19eff9e
# ╠═ff9d8e91-c9c7-4257-b4f6-ef62bbf89cb8
# ╠═c7489681-213f-48db-94f1-f0b670f10350
# ╠═fcdcae67-5cbd-432b-8878-ab7ca7cd728a
# ╠═0b37c368-ac33-451c-a75c-bf333ce33e60
# ╟─bbae926c-1b3e-4261-86f0-68be430cdec5
# ╠═d734a6d6-7284-4844-909c-23a3733b3bc7
# ╠═a23245ac-01fb-4b6b-8858-77e6d8483e6e
# ╠═115e3ddc-5b7a-4f66-b88a-3ce61e7005a1
# ╟─d5390279-fbe1-42d8-9cfd-84f7e74fd34d
# ╠═cb90ded3-e7d8-4af3-ade7-6794317c3f93
