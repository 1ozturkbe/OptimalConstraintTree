
# last_regr = bbr.learners[end]
# df = DataFrame([Float64 for i in 1:length(bbr.vars)], string.(bbr.vars))
# subthres_idx = findall(y -> y <= threshold, bbr.Y)
# subthres_leaves = IAI.apply(last_regr, bbr.X[subthres_idx,:])
# bbr.knn_tree = KDTree(OCT.normalized_data(bbr)', reorder = false)
# for leaf in unique(subthres_leaves)
#     idx_in_leaf = [i for i = 1:length(subthres_idx) if subthres_leaves[i] == leaf]
#     maxes = maximum.(eachcol(bbr.X[idx_in_leaf,:]))
#     mins = minimum.(eachcol(bbr.X[idx_in_leaf,:]))

#     idx_just_out = [] 
#     for i in idxs_in_leaf
        
#     = unique(hcat(idxs[i]), for i in idx_in_leaf)

# function neldermead_sample(bbr:BlackBoxRegressor, X::DataFrame, Y::Array, threshold; k::Int64 = length(bbr.vars))
#     vks = string.(bbl.vars)
#     df = DataFrame([Float64 for i in vks], vks)
#     build_knn_tree(bbl);
#     idxs, dists = find_knn(bbl, k=k);
#     positives = findall(x -> x .>= 0 , bbl.Y);
#     feas_class = classify_patches(bbl, idxs);
#     for center_node in positives # This loop is for making sure that every possible root is sampled only once.
#         if feas_class[center_node] == "mixed"
#             nodes = [idx for idx in idxs[center_node] if bbl.Y[idx] < 0];
#             push!(nodes, center_node)
#             np = secant_method(bbl.X[nodes, :], bbl.Y[nodes, :])
#             append!(df, np);
#         end
#     end
#     return df
# end