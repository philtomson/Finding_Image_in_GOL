using Base.Threads
using LinearAlgebra
using BenchmarkTools
using DSP
using Images
using Ditherings
using DSP
using JLD2
using Flux

batch_size = 50
MAX_MUTATIONS = 6
n_generations = 6

kernel =       UInt8.([ 1 1 1   #kernel for GoL
                        1 0 1
                        1 1 1 ])

kernel = reshape(kernel, 3,3,1, :)

liveordie(count, current) = 
(current == 0 && count==3) || (current == 1 && (1 < count <=3))

function mutate!(imgs, kw=0, max_mutations = MAX_MUTATIONS)
	num_imgs = size(imgs)[1]
	img_size = size(imgs[1])
	for r in 1:num_imgs
		num_mutations = rand(1:max_mutations)
	    for i in 1:num_mutations #try 3 random pixels (TODO make this paramatizable)
	       x,y = rand(1:img_size[1]-kw), rand(1:img_size[2]-kw)
         imgs[r][x,y] ⊻= 0x01 #flip random bits in image
      end
  end
end

cmodl = Chain(
      Conv((3,3), 1=>1, pad=(1,1))
      #eventually make the liveordie layer here
      )

 #Now we have to set up our weights kernel for GOL:
Flux.params(cmodl)[1][1,1] = 1.0
Flux.params(cmodl)[1][1,2] = 1.0
Flux.params(cmodl)[1][1,3] = 1.0
Flux.params(cmodl)[1][2,1] = 1.0
Flux.params(cmodl)[1][2,2] = 0.0
Flux.params(cmodl)[1][2,3] = 1.0
Flux.params(cmodl)[1][3,1] = 1.0
Flux.params(cmodl)[1][3,2] = 1.0
Flux.params(cmodl)[1][3,3] = 1.0


monalisa_img = mktemp() do fn,f
   download("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/483px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg", fn)
   load(fn)
end



monalisa_img_gs = Gray.(monalisa_img)
monalisa_img_dithered = Ditherings.FloydSteinbergDither4Sample(monalisa_img_gs, Ditherings.ZeroOne)
lisa_contrast_img = map(x -> round(x), Gray.(monalisa_img_dithered))
lisa_img = Gray.(lisa_contrast_img) #Gray.(Int8.(lisa_contrast_img))
lisas = repeat(Float32.(lisa_img), 1,1, batch_size)
lisatrain = reshape(lisas, 720, 483, 1, :)
lisa_loaf = UInt8.(lisatrain)
 #cmodl(lisatrain)
canvas_loaf = UInt8.(round.(rand(Float32, size(lisatrain))))
cmodl(canvas_loaf) #works

   
function tileimg(img::AbstractArray, num::Integer)
	outarray = Array{typeof(img)}(undef, num)
	for i in 1:num
		outarray[i] = copy(img)
	end
	outarray
end


width,height = size(lisa_img)
 #lisa_loaf = tileimg(UInt8.(lisa_img), batch_size)


liveordie(count::UInt8, current::UInt8) = 
(current == 0x00 && count==0x03) || (current == 0x01 && (0x01 < count <=0x03))

function convGOL_flux(A::AbstractArray, kern::AbstractArray)
	counts = Flux.conv(A, kern; pad=1)
	out    = zeros(eltype(A), size(A))
	#size(counts) should == size(A)
	Threads.@threads for i in CartesianIndices(size(A))
		out[i] = liveordie(UInt8(counts[i]), A[i])
	end
	out
end

outconv = Flux.conv(canvas_loaf, kernel, pad=1)

convGOL_flux(canvas_loaf, kernel)

function convGOL_conv(A::AbstractMatrix, kern::AbstractMatrix)
	counts = conv(A, kern)[2:end-1, 2:end-1]
	out    = zeros(eltype(A), size(A))
	#size(counts) should == size(A)
	@inbounds for i in CartesianIndices(size(A))
		out[i] = liveordie(UInt8(counts[i]), A[i])
	end
	out
end

function convGOL_dot(A::AbstractMatrix, kern::AbstractMatrix)
	b = CartesianIndex(1,1)
	out = similar(A)
	@inbounds for i in CartesianIndices(A)[2:end-1, 2:end-1]
		count = dot(view(A,i-b:i+b), kern)
		out[i] = liveordie(count, A[i])
	end
	out
end

convGOL = convGOL_flux

function GOLsteps(state, kern, ngens::Integer)
	for _ in 1:ngens
		state = convGOL(state, kern)
	end
	state
end


GOLsteps(canvas_loaf, kernel, n_generations )

function convolveimgs(images, kern::AbstractMatrix, ngens::Integer)
          num_imgs = size(images)[1]
          out = similar(images)
          for img_num in 1:num_imgs
             #out[img_num] = conv(images[img_num], kern)
             out[img_num] = GOLsteps(images[img_num], kern, ngens)
          end
          out
       end

function convolveimgs_t(images, kern::AbstractMatrix, ngens::Integer)
          num_imgs = size(images)[1]
          out = similar(images)
          Threads.@threads for img_num in 1:num_imgs
             #out[img_num] = conv(images[img_num], kern)
             out[img_num] = GOLsteps(images[img_num], kern, ngens)
          end
          out
       end

function tilerand_prev(sz, num::Integer)
   outarray = []
   for i in 1:num
      push!(outarray, rand(Float64, sz))
   end
   outarray
end

function tilerand(img, num::Integer) #will return a matrix like img
	outarray = Array{typeof(img)}(undef, num)
	tmp = [] #Array{Float32,2}(undef, num)
	for i in 1:num
		push!(tmp, eltype(img).(map( x-> round.(x), rand(Float32, size(img)))))
		outarray[i] = eltype(img).(map( x-> round.(x), rand(Float32, size(img))))
	end
	#eltype(img).(map(x -> round.(x), tmp))
	outarray
end	

 #rand_imgs = tilerand((720,483), 50)




 #@benchmark convolveimgs(rand_imgs, kernel)
 #@benchmark convolveimgs_t(rand_imgs, kernel)

 #@benchmark convolveimgs(lisa_loaf, kernel, n_generations)
 #@benchmark convolveimgs_t(lisa_loaf, kernel, n_generations)

#Now we need the mutator tensor (terminology from the article)
#Same shape as canvas_loaf, but all zeros execpt for one 1 in each "slice"
function mutate(imgs::AbstractArray, kw=0, max_mutations = MAX_MUTATIONS)
	num_imgs = size(imgs)[end]
	img_size = size(imgs[:,:,1,1])
	outarray = zeros(eltype(imgs), size(imgs))

	for r in 1:num_imgs
		num_mutations = rand(1:max_mutations)
	    for i in 1:num_mutations #try 3 random pixels (TODO make this paramatizable)
	       x,y = rand(1:img_size[1]-kw), rand(1:img_size[2]-kw)
	       outarray[x,y,1,r] = 0x01
		   end
    end
	outarray
end

#possibly an alternative to rmse for loss
function img_diff(a::AbstractArray, b::AbstractArray)
	sum(xor.(a,b))
end

function imgs_diff(a, b)
   @assert size(a)[end] == size(b)[end]
   errors = []
   num_imgs = size(a)[end]
   for img in 1 : num_imgs
      push!(errors, sum(xor.(a[:,:,1,img], b[:,:,1,img])))
   end
   errors
end

function hill_climb(original, canvas, iterations, kern=kernel, num_gens=n_generations)
	best_score  = Inf
	best_canvas = copy(canvas)
	s_canvas    = copy(canvas)
	fitness_progress = []
	for run in 1:iterations
    s_canvas = [xor.(a,b) for (a,b) in zip(s_canvas, mutate(canvas, size(kern)[1]))]
    after_canvas = GOLsteps(s_canvas, kern, num_gens)
		#rmse_vals = rmse.(original, s_canvas)
		rmse_vals = imgs_diff(original, after_canvas)
		curr_min  = minimum(rmse_vals)
		push!(fitness_progress, curr_min) 
		if curr_min < best_score
			best_score = curr_min
      best_canvas = repeat(s_canvas[:,:,1,argmin(rmse_vals)],1,1,1,size(s_canvas)[end])
      #TODO: best_canvas = reshape(best_canvas, 720, 483, 1, :)
		end
		s_canvas = best_canvas
	end
	return s_canvas, fitness_progress
end


