using MLDataUtils
using Random

function prepare_iris()
    X, Y, fnames = load_iris(150);
    Y = reshape(Y, length(Y), 1)
    iris_dataset = hcat(transpose(X), Y)
    iris_dataset[shuffle(1:end), :]
end

function split_dataset(ds, val)
    if val > 1.0
        return "Value should be from a range [0, 1]!"
    end
    train_len = convert(UInt64, round(val * length(ds[:,1])))
    train_set = ds[1:train_len,:]
    test_set = ds[train_len+1:length(ds[:,1]),:]
    tuple(train_set, test_set)
end

function adjust_dataset(ds)
    dataset = ds[:,1:length(ds[1,:])-1]
    categories = ds[:,length(ds[1,:])]
    expected_values = zeros(length(ds[:,1]), 3)
    for (idx, val) in enumerate(categories)
        if val == "setosa"
            setindex!(expected_values, 1.0, idx, 1)
        elseif val == "versicolor"
            setindex!(expected_values, 1.0, idx, 2)
        elseif val == "virginica"
            setindex!(expected_values, 1.0, idx, 3)
        end
    end
    tuple(convert(Matrix{Float64}, dataset), expected_values)
end
