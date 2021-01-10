// 多分类问题拟合
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getIrisData, IRIS_CLASSES } from './data';

window.onload = async () => {
    // 引入Iris训练集和验证集
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);
    // 定义带有softmax的多层神经网络模型
    const model = tf.sequential();
    // 添加两层神经元网络
    // 设计层的神经元个数, inputShape和激活函数
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [xTrain.shape[1]],
        // 非线性激活函数即可
        activation: 'sigmoid'
    }));
    model.add(tf.layers.dense({
        // 神经元个数必须是输出类别的个数
        units: 3,
        // 多分类激活函数"softmax", 保证输出的值概率加起来为1
        activation: 'softmax'
    }));

    model.compile({
        // 交叉熵损失函数: 对数损失函数多分类版本
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(0.1),
        // 准确度的度量
        metrics: ['accuracy']
    });

    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest],
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    window.predict = (form) => {
        const input = tf.tensor([[
            form.a.value * 1,
            form.b.value * 1,
            form.c.value * 1,
            form.d.value * 1,
        ]]);
        const pred = model.predict(input);
        alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`);
    };
};
