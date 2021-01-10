import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data.js';

window.onload = async () => {
    // 加载训练的数据
    const data = getData(400);
    // 加载数据可视化
    tfvis.render.scatterplot(
        { name: 'XOR 训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    );
    // 初始化一个神经网络模型
    const model = tf.sequential();
    // 为神经网络模型添加两个层: 隐藏层和输出层, 设计层的神经元个数/inputShape/激活函数(让神经元网络具有非线性的能力)
    // 隐藏层
    model.add(tf.layers.dense({
        units: 4,
        inputShape: [2],
        // Relu激活函数用于隐层神经元输出。
        activation: 'relu'
    }));
    // 输出层
    model.add(tf.layers.dense({
        units: 1,
        // Sigmoid函数是一个在生物学中常见的S型函数，也称为S型生长曲线。
        // 在信息科学中，由于其单增以及反函数单增等性质，Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间
        activation: 'sigmoid'
    }));
    // 损失函数和优化器
    model.compile({
        // 损失函数
        loss: tf.losses.logLoss,
        // 优化器: adam可自行调节学习速率
        optimizer: tf.train.adam(0.1)
    });
    // 输入数据
    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    // 正确的输出数据
    const labels = tf.tensor(data.map(p => p.label));
    // 数据拟合并可视化
    await model.fit(inputs, labels, {
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss']
        )
    });

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]));
        alert(`预测结果：${pred.dataSync()[0]}`);
    };
};
