分类:
    class 文章
    {
        情感 = 文章情感(正确答案)
        计算结果 = 文章情感(KNN结果)
        距离 = 与某篇文章距离

        getKnn(训练集, k)
        {
            for 文章 in 训练集
                文章.距离 = 两者之间距离(欧几里德，曼哈顿或余弦角)
            sort(训练集)
            for 文章 in 训练集[0:k-1]
                统计情感数量
            self.情感 = 前K文章中最多的情感
        {
    }

    getAllKnn(测试集，训练集，k)
    {
        正确数 = 0
        错误数 = 0

        for 文章 in 测试集:
            文章.getAllKnn(训练集,k)
            if 文章.情感 = 文章.计算结果:
                正确数 + 1
            else:
                错误数 + 1
    }

    Main()
    {
        readWord();
        getAllKnn(测试集,训练集, k)
    }
    


回归:
    class 文章
    {
        情感s = 文章情感s(从文件中读取的情感概率)
        计算结果s = 计算的出的情感概率
        距离 = 与某篇文章的距离

        getKnn(训练集, k)
        {
            for 文章 in 训练集:
                文章.距离 = 两者之间的距离(欧几里德，曼哈顿或余弦角)
                for 情感 in 文章.情感s:
                    self.情感s[情感] += 文章.情感/文章.距离

            对self.情感s归一化    
        }
    }

    getAllKnn(测试集，训练集，k)
    {
        for 文章 in 测试集:
            文章.getAllKnn(训练集,k)
        
        将结果写入csv文件中
    }

    Main()
    {
        readWord();
        getAllKnn(测试集,训练集, k)
    }