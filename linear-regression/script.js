import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

window.onload = async () => {
    const xs = [1, 2, 3, 4];
    const ys = [1, 3, 5, 7];
    // 将输入数据可视化
    tfvis.render.scatterplot(
        { name: '线性回归训练集' },
        { values: xs.map((x, i) => ({ x, y: ys[i] })) },
        { xAxisDomain: [0, 5], yAxisDomain: [0, 8] }
    );
    // 初始化一个神经元模型, 创建一个连续的模型(这一层的输入一定是上一层的输出)
    const model = tf.sequential();
    // 设置神经元个数和tensor形状(全连接层 )
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    // 设置损失函数和优化器(学习速率0.1)
    model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) });
    // 将数据转换为tensor
    const inputs = tf.tensor(xs);
    // 正确的标签需要放进去
    const labels = tf.tensor(ys);
    // 异步训练模型
    await model.fit(inputs, labels, {
        // 小批量样本的数据量
        batchSize: 4,
        // 迭代整个数据的训练次数
        epochs: 100,
        // 训练过程数据可视化
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练过程' },
            ['loss']
        )
    });

    const output = model.predict(tf.tensor([5]));
    alert(`如果 x 为 5，那么预测 y 为 ${output.dataSync()[0]}`);
};