module Tiling3D

export tile_array, dmem, Tiling, tile_pointer, tiles_array

# Dummy dmem function for demonstration.
# For a 3D tile, dims is a tuple of three integers.
function dmem(ptr::Ptr{T}, dims::NTuple{3,Int}) where T
    println("Creating DMEM with pointer: ", ptr, " and dimensions: ", dims)
    # Replace the println above with the actual DMEM object creation.
    return nothing
end

struct Tile3D{T, M<:AbstractArray{T,3}}
    #view::SubArray{T,3,M,NTuple{3,UnitRange{Int}},false}
    view::SubArray{T,3,M,Tuple{UnitRange{Int},UnitRange{Int},UnitRange{Int}},false}
    tile_i::Int
    tile_j::Int
    tile_k::Int
end

# Function to split a 3D array into tiles.
# tile_array(A, tile_dim1, tile_dim2, tile_dim3)
#   - A: 3D array.
#   - tile_dim1, tile_dim2, tile_dim3: desired tile dimensions.
# Returns a vector of SubArray views into A.
function tile_array(A::AbstractArray{T,3}, tile_dim1::Int, tile_dim2::Int, tile_dim3::Int, vector=true) where T
    n1, n2, n3 = size(A)
    tiles = Vector{Tile3D{eltype(A), typeof(A)}}()
    #tiles = Vector{SubArray{T,3,typeof(A),Tuple{UnitRange{Int}, UnitRange{Int}, UnitRange{Int}},false}}()
    tile_i = 0
    tile_j = 0
    tile_k = 0
    for i in 1:tile_dim1:n1
        tile_j = 0
        tile_i += 1
        for j in 1:tile_dim2:n2
            tile_k = 0
            tile_j += 1
            for k in 1:tile_dim3:n3
                tile_k += 1
                i_end = min(i + tile_dim1 - 1, n1)
                j_end = min(j + tile_dim2 - 1, n2)
                k_end = min(k + tile_dim3 - 1, n3)
                view_tile = @view A[i:i_end, j:j_end, k:k_end]
                ptr = pointer(view_tile)
                push!(tiles, Tile3D(view_tile, tile_i, tile_j, tile_k))
            end
        end
    end
    if vector
        return tiles
    else
        return reshape(tiles, tile_i, tile_j, tile_k)
    end
end

# Tiling struct for 3D arrays.
# Fields:
#   - A: the original 3D array.
#   - tile_dim1, tile_dim2, tile_dim3: the dimensions for each tile.
#   - tiles: a vector of tile views.
#   - n_tile_dim1, n_tile_dim2, n_tile_dim3: number of tiles in each dimension.
struct Tiling{T, M<:AbstractArray{T,3}}
    A::M
    tile_dim1::Int
    tile_dim2::Int
    tile_dim3::Int
    tiles::Vector{Tile3D{T, M}}
    n_tile_dim1::Int
    n_tile_dim2::Int
    n_tile_dim3::Int
end

#Tiling(A, tile_dim1, tile_dim2, tile_dim3)
#    Construct a Tiling object from a 3D array `A` by splitting it into tiles of size
#    `tile_dim1 × tile_dim2 × tile_dim3`. Tiles at the boundaries may be smaller if A's
#    dimensions are not exact multiples of the tile sizes.
function Tiling(A::AbstractArray{T,3}, tile_dim1::Int, tile_dim2::Int, tile_dim3::Int) where T
    tiles = tile_array(A, tile_dim1, tile_dim2, tile_dim3)
    n1, n2, n3 = size(A)
    n_tile_dim1 = cld(n1, tile_dim1)
    n_tile_dim2 = cld(n2, tile_dim2)
    n_tile_dim3 = cld(n3, tile_dim3)
    return Tiling{T, typeof(A)}(A, tile_dim1, tile_dim2, tile_dim3, tiles,
                                                n_tile_dim1, n_tile_dim2, n_tile_dim3)
end

# Helper functions for pointer computation.
# Recursively unwrap a reshaped array to get the underlying contiguous array.
function underlying_array(A::AbstractArray)
    if A isa Base.ReshapedArray
            return underlying_array(A.parent)
    elseif A isa UnitRange
            # Converting a UnitRange to an array (note: this makes a copy)
            return collect(A)
    else
            return A
    end
end

# Compute a pointer to the first element of a tile.
function tile_pointer(tile::Tile3D)
    inds = ntuple(i -> first(tile.view.indices[i]), ndims(tile.view))
    li = LinearIndices(tile.view.parent)[inds...]
    uA = underlying_array(tile.view.parent)
    return pointer(uA, li)
end

# Utility to reshape the vector of tiles into a 3D array.
# This allows you to iterate over the tiles in a structured (n_tile_dim1 × n_tile_dim2 × n_tile_dim3)
# fashion.
function tiles_array(t::Tiling)
    return reshape(t.tiles, (t.n_tile_dim1, t.n_tile_dim2, t.n_tile_dim3))
end

end # module Tiling3D

"""
    @tiling3d data=<A or (A, B, ...)> [tile_dim1=<n1>] [tile_dim2=<n2>] [tile_dim3=<n3>] [mode=<:iterate|:list>] [begin ... end]

    Creates a tiling of a 3D array (or tuple of 3D arrays) and iterates over the tiles,
    executing the provided block for each tile (or tuple of tiles if multiple arrays are given).
    If no block is provided, the macro returns a tuple of iterators (one per input array).

    Inside the block, the variable `tile` is bound to:
    - a single Tile3D (if one array was provided), or
    - a tuple of Tile3D objects (if several arrays were provided).
    """
macro tiling3d(args...)
    # Set default keyword values.
    local data_expr = nothing
    local tile_dim1_expr = :(2)
    local tile_dim2_expr = :(2)
    local tile_dim3_expr = :(2)
    local mode_expr = :(iterate)
    local block_expr = nothing

    # Parse the macro arguments.
    for arg in args
        if arg isa Expr && arg.head == :(=)
            if arg.args[1] == :data
                data_expr = arg.args[2]
            elseif arg.args[1] == :tile_dim1
                tile_dim1_expr = arg.args[2]
            elseif arg.args[1] == :tile_dim2
                tile_dim2_expr = arg.args[2]
            elseif arg.args[1] == :tile_dim3
                tile_dim3_expr = arg.args[2]
            elseif arg.args[1] == :mode
                mode_expr = arg.args[2]
            else
                error("Unknown keyword: ", arg.args[1])
            end
        else
            block_expr = arg
        end
    end

    if data_expr === nothing
        error("Missing keyword argument: data")
    end

    # If no block is provided, return the tuple of iterators.
    if block_expr === nothing
        return esc(quote
            local __data = $data_expr
            local __tilings = if __data isa Tuple
                map(x -> Tiling3D.Tiling(x, $tile_dim1_expr, $tile_dim2_expr, $tile_dim3_expr), __data)
            else
                (Tiling3D.Tiling(__data, $tile_dim1_expr, $tile_dim2_expr, $tile_dim3_expr),)
            end
            # For 3D tiling, we simply use the tiles vector as the iterator.
            local _it = $iterator_expr
            local __iters = _it isa Tuple ?
                map((tiling, order) ->
                    order == :row_major ? Tiling3D.row_major_tiles(tiling) : Tiling3D.col_major_tiles(tiling),
                    __tilings, _it) :
                map(t -> ($iterator_expr == :row_major ? Tiling3D.row_major_tiles(t) : Tiling3D.col_major_tiles(t)), __tilings)
            __iters
        end)
    else
        return esc(quote
            println("DEBUG: block_expr is ", $(string(block_expr)))
            local __data = $data_expr
            local __tilings = if __data isa Tuple
                map(x -> Tiling3D.Tiling(x, $tile_dim1_expr, $tile_dim2_expr, $tile_dim3_expr), __data)
            else
                (Tiling3D.Tiling(__data, $tile_dim1_expr, $tile_dim2_expr, $tile_dim3_expr),)
            end
            local _it = $iterator_expr
            local __iters = _it isa Tuple ?
                map((tiling, order) ->
                    order == :row_major ? Tiling3D.row_major_tiles(tiling) : Tiling3D.col_major_tiles(tiling),
                    __tilings, _it) :
                map(t -> ($iterator_expr == :row_major ? Tiling3D.row_major_tiles(t) : Tiling3D.col_major_tiles(t)), __tilings)
            for __tile in zip(__iters...)
                let tile = __tile
                    $block_expr
                end
            end
        end)
    end
end
