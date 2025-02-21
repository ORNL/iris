module Tiling2D

export tile_array, tile_row_iterator, tile_col_iterator, Tiling, row_tiles, col_tiles, tile_pointer

struct Tile2D{T, M<:AbstractMatrix{T}}
    view::SubArray{T,2,M,Tuple{UnitRange{Int},UnitRange{Int}},false}
    dmem::Any
    row_index::Int
    col_index::Int
end

# Function to split a 2D array into tiles.
# tile_array(A, tile_rows, tile_cols)
#   - A: 2D array (matrix)
#   - tile_rows, tile_cols: dimensions of each tile.
# Returns a vector of views (subarrays) into A.
function tile_array(A::AbstractMatrix, tile_rows::Int, tile_cols::Int, vector=true)
    nrows, ncols = size(A)
    tiles = Vector{Tile2D{eltype(A), typeof(A)}}()
    #tiles = Vector{SubArray{eltype(A),2,typeof(A),Tuple{UnitRange{Int},UnitRange{Int}},false}}()
    row_index = 0
    col_index = 0
    for i in 1:tile_rows:nrows
        row_index += 1
        col_index = 0
        for j in 1:tile_cols:ncols
            col_index += 1
            row_end = min(i + tile_rows - 1, nrows)
            col_end = min(j + tile_cols - 1, ncols)
            view_tile = @view A[i:row_end, j:col_end]
            dev_size = [tile_cols, tile_rows]
            offset = [j-1, i-1]
            #ptr = pointer(view_tile)
            mem = nothing
            if isdefined(Main, :IrisHRT)
                mem = Main.IrisHRT.dmem_offset(A, dev_size, offset)
            end
            push!(tiles, Tile2D(view_tile, mem, row_index, col_index))
        end
    end
    if vector
        return tiles
    else
        return reshape(tiles, row_index, col_index)
    end
end

# Iterators for the tiles.
# Given a vector of tiles and the number of tiles per row (num_tile_cols),
# these functions return iterators that yield a row or a column (as a vector) of tiles.
function tile_row_iterator(tiles::Vector, num_tile_cols::Int)
    nrows = length(tiles) ÷ num_tile_cols
    return ( tiles[(r-1)*num_tile_cols+1 : r*num_tile_cols] for r in 1:nrows )
end

function tile_col_iterator(tiles::Vector, num_tile_cols::Int)
    nrows = length(tiles) ÷ num_tile_cols
    return ( [ tiles[(r-1)*num_tile_cols+c] for r in 1:nrows ] for c in 1:num_tile_cols )
end

# A struct to package the tiling of a 2D array.
# Fields:
#   - A: the original 2D array
#   - tile_rows, tile_cols: dimensions for each tile
#   - tiles: a vector of tile views
#   - n_tile_rows, n_tile_cols: the number of tile rows and columns
struct Tiling{T, M<:AbstractMatrix{T}}
    A::M
    tile_rows::Int
    tile_cols::Int
    tiles::Vector{Tile2D{T, M}}
    n_tile_rows::Int
    n_tile_cols::Int
end

# Tiling(A, tile_rows, tile_cols)
#    Create a Tiling object from a 2D array `A` by splitting it into tiles of size
#    `tile_rows`×`tile_cols`. The last tiles in a row or column may be smaller if A’s
#    dimensions are not multiples of the tile size.
function Tiling(A::AbstractMatrix, tile_rows::Int, tile_cols::Int)
    tiles = tile_array(A, tile_rows, tile_cols)
    nrows, ncols = size(A)
    n_tile_rows = cld(nrows, tile_rows)
    n_tile_cols = cld(ncols, tile_cols)
    return Tiling{eltype(A), typeof(A)}(A, tile_rows, tile_cols, tiles, n_tile_rows, n_tile_cols)
end


#    row_tiles(t::Tiling)
# Return an iterator over the rows of tiles. Each element is a vector containing
# the tiles in one row.
function row_tiles(t::Tiling)
    return tile_row_iterator(t.tiles, t.n_tile_cols)
end

#    col_tiles(t::Tiling)
# Return an iterator over the columns of tiles. Each element is a vector containing
# the tiles in one column.
function col_tiles(t::Tiling)
    return tile_col_iterator(t.tiles, t.n_tile_cols)
end

# Helper functions to obtain a pointer for a contiguous tile.
# Unwrap a reshaped array to get the underlying contiguous array.
function underlying_array(A::AbstractArray)
    if A isa Base.ReshapedArray
            return underlying_array(A.parent)
    elseif A isa UnitRange
            # Convert the UnitRange to an Array
            return collect(A)
    else
            return A
    end
end

# Compute a pointer to the first element of a tile.
function tile_pointer(tile::Tile2D)
    # Get the first index in each dimension for the tile's view.
    inds = ntuple(i -> first(tile.view.indices[i]), ndims(tile.view))
    # Compute the corresponding linear index in the tile's parent array.
    li = LinearIndices(tile.view.parent)[inds...]
    # Unwrap the parent to ensure we have an array that supports pointer.
    uA = underlying_array(tile.view.parent)
    return pointer(uA, li)
end

end  # module Tiling2D

"""
@tiling2d data=<A or (A, B, ...)> iterator=<:row_major|:col_major> [tile_rows=<n>] [tile_cols=<m>] [mode=<:iterate|:list>] begin
<body>
end

When mode is set to `:iterate` (the default), the macro performs the iteration over the
tiling(s) and executes the provided block for each group of corresponding tiles.
When mode is set to `:list`, the macro returns a tuple of iterators (one for each input array).

Keyword arguments:
- `data`: a 2D array or a tuple of 2D arrays.
- `iterator`: either `:row_major` or `:col_major`.
- `tile_rows` (optional): number of rows per tile (default is 64).
- `tile_cols` (optional): number of columns per tile (default is 64).
- `mode` (optional): either `:iterate` (default) or `:list`.

Inside the loop (in iteration mode), the variable `tile` is bound to:
- a single Tile2D (if one array was provided), or
- a tuple of Tile2D objects (if several arrays were provided).
"""

    
macro tiling2d(args...)
    # Parse the arguments: expect keyword arguments and optionally a block.
    local data_expr      = nothing
    local iterator_expr  = nothing
    local tile_rows_expr = :(2)
    local tile_cols_expr = :(2)
    local mode_expr      = :(iterate)  # default mode is iteration
    local block_expr     = nothing
    local name_expr      = :(tile)

    # Process each argument.
    for arg in args
        if arg isa Expr && arg.head == :(=)
            # Keyword argument of the form key = value.
            if arg.args[1] == :data
                data_expr = arg.args[2]
            elseif arg.args[1] == :iterator
                iterator_expr = arg.args[2]
            elseif arg.args[1] == :tile_rows
                tile_rows_expr = arg.args[2]
            elseif arg.args[1] == :tile_cols
                tile_cols_expr = arg.args[2]
            elseif arg.args[1] == :mode
                mode_expr = arg.args[2]
            elseif arg.args[1] == :name
                name_expr = arg.args[2]
            else
                error("Unknown keyword: ", arg.args[1])
            end
        else
            # Assume this is the loop body block.
            block_expr = arg
        end
    end

    if data_expr === nothing
        error("Missing keyword argument: data")
    end
    if iterator_expr === nothing
        error("Missing keyword argument: iterator")
    end

    if block_expr === nothing
        return esc(quote
            local __data = $data_expr
            local __tilings = if __data isa Tuple
                map(x -> Tiling2D.Tiling(x, $tile_rows_expr, $tile_cols_expr), __data)
            else
                (Tiling2D.Tiling(__data, $tile_rows_expr, $tile_cols_expr),)
            end
            # Allow the iterator argument to be either a symbol or a tuple.
            local _it = $iterator_expr
            local __iters = _it isa Tuple ?
                map((tiling, order) ->
                                order == :row_major ? Tiling2D.row_tiles(tiling) : Tiling2D.col_tiles(tiling),
                                __tilings, _it) :
                map(t -> (_it == :row_major ? Tiling2D.row_tiles(t) : Tiling2D.col_tiles(t)), __tilings)
            __iters
        end)
    else
        return esc(quote
            # (Optional) Debug print.
            local __data = $data_expr
            local __tilings = if __data isa Tuple
                map(x -> Tiling2D.Tiling(x, $tile_rows_expr, $tile_cols_expr), __data)
            else
                (Tiling2D.Tiling(__data, $tile_rows_expr, $tile_cols_expr),)
            end
            local _it = $iterator_expr
            local __iters = _it isa Tuple ?
                map((tiling, order) ->
                                order == :row_major ? Tiling2D.row_tiles(tiling) : Tiling2D.col_tiles(tiling),
                                __tilings, _it) :
                map(t -> (_it == :row_major ? Tiling2D.row_tiles(t) : Tiling2D.col_tiles(t)), __tilings)
            # In iteration mode, zip the iterators and run the block.
            for __rows in zip(__iters...)
                for __tile in zip(__rows...)
                    let $name_expr = __tile
                        $block_expr
                    end
                end
            end
       end)
    end
end
 
macro tiling2d_group(args...)
    # Parse macro arguments
    local data_expr      = nothing
    local tile_rows_expr = :(2)
    local tile_cols_expr = :(2)
    local block_expr     = nothing
    local group_expr     = nothing
    local group_name_expr = :(group)
    for arg in args
        if arg isa Expr && arg.head == :(=)
            if arg.args[1] == :data
                data_expr = arg.args[2]
            elseif arg.args[1] == :group
                group_expr = arg.args[2]
            elseif arg.args[1] == :tile_rows
                tile_rows_expr = arg.args[2]
            elseif arg.args[1] == :tile_cols
                tile_cols_expr = arg.args[2]
            elseif arg.args[1] == :group_name
                group_name_expr = arg.args[2]
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
    if group_expr === nothing
        error("Missing keyword argument: group")
    end

    # If no block is provided, return the groups as an array.
    if block_expr === nothing
        return esc(quote
            local __data = $data_expr
            local __tiling = Tiling2D.Tiling(__data, $tile_rows_expr, $tile_cols_expr)
            local __tile_matrix = reshape(__tiling.tiles, __tiling.n_tile_rows, __tiling.n_tile_cols)
            if $group_expr == :row
                collect(eachrow(__tile_matrix))
            elseif $group_expr == :col
                collect(eachcol(__tile_matrix))
            else
                error("Invalid group type: ", $group_expr)
            end
        end)
    else
        # In iteration mode, bind the group to the given name in the block.
        return esc(quote
            local __data = $data_expr
            local __tiling = Tiling2D.Tiling(__data, $tile_rows_expr, $tile_cols_expr)
            local __tile_matrix = reshape(__tiling.tiles, __tiling.n_tile_rows, __tiling.n_tile_cols)

            if $group_expr == :row
                for (__index, __group) in enumerate(eachrow(__tile_matrix))
                    let $group_name_expr = (index=__index, tiles=__group)
                        $block_expr
                    end
                end
            elseif $group_expr == :col
                for (__index, __group) in enumerate(eachcol(__tile_matrix))
                    let $group_name_expr = (index=__index, tiles=__group)
                        $block_expr
                    end
                end
            else
                error("Invalid group type: ", $group_expr)
            end
        end)
    end
end
