function [W, S, labels, C] = SPC(Lraw, lambda, alpha, T)

    [m, n] = size(Lraw);


    C = max(Lraw(:)); 
    if C <= 0
        error('数据中没有有效类别标签（全是0）');
    end
    if any(Lraw(:) < 0) || any(Lraw(:) > C)
        error('检测到非法标签值，请检查数据！');
    end


    Ycell = cell(m,1);
    for i = 1:m
        Yi = zeros(n, C);
        for j = 1:n
            if Lraw(i,j) ~= 0   
                c = Lraw(i,j);
                Yi(j, c) = 1;
            end
        end
        Ycell{i} = Yi;
    end


    total_sum = 0;
    for i = 1:m
        total_sum = total_sum + sum(Ycell{i}(:));
    end
    if total_sum == 0
        error('所有工人标注矩阵全为 0，请检查输入数据！');
    end


    z0 = zeros(n, 1); 
    for j = 1:n
        votes = Lraw(:, j);
        votes = votes(votes > 0); 
        if ~isempty(votes)
            z0(j) = mode(votes);
        else
            z0(j) = randi(C);
        end
    end

    W = zeros(m, C);
    for i = 1:m
        for c = 1:C
            idx = (z0 == c); 
            Yi_c = (Lraw(i, :)' == c);
            valid = (Lraw(i, :)' > 0);
            if sum(valid & idx) > 0
                W(i, c) = sum(Yi_c & idx) / sum(valid & idx);
            else
                W(i, c) = 0;
            end
        end
    end

    for t = 1:max(1, T)
        for c = 1:C
            Yc = zeros(n, m);
            for i = 1:m
                Yc(:, i) = Ycell{i}(:, c);
            end

            if alpha > 0 && t > 1
                res = zeros(n,1);  
                for j = 1:n
                    Sj = S(j,:);   
                    tmp = 0;
                    for i = 1:m
                        Yij = Ycell{i}(j,:); 
                        tmp = tmp + sum((Yij - Sj).^2); 
                    end
                    res(j) = tmp; 
                end 
                alpha_t = prctile(res,(100/T)*t);  
                v = max(0, min(1, 1 - res ./ alpha_t));
            else
                v = ones(n,1);
            end

            V = diag(v);
            Gc = Yc' * V * Yc;
            bc = Gc * ones(m, 1);

            W(:, c) = (m * Gc + lambda * eye(m)) \ bc;


            W(:, c) = max(W(:, c), 0);
        end

        S = zeros(n, C);
        for c = 1:C
            Yc = zeros(n, m);
            for i = 1:m
                Yc(:, i) = Ycell{i}(:, c);
            end
            S(:, c) = Yc * W(:, c);
        end
    end

    [~, labels] = max(S, [], 2);

end