import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistData } from './data';

window.onload = async () => {
    const data = new MnistData();
    await data.load();
    const examples = data.nextTestBatch(20);
    const surface = tfvis.visor().surface({ name: '输入示例' });
    // 将数据转化为图片
    for (let i = 0; i < 20; i += 1) {
        // 提取每个数据的Tensor, tidy()方法清除内存中多余的数据, 防止内存泄漏
        const imageTensor = tf.tidy(() => {
            return examples.xs
                .slice([i, 0], [1, 784])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);
    }
    /* 卷积层: 使用多个卷积核对图像进行卷积操作提取特征
        卷积层有权重需要训练, 卷积核就是权重
        池化层: 提取最强体征, 扩大感受野, 减少计算量, 池化层没有权重需要训练
        全连接层: 作为输出层, 作为分类器, 有权重需要训练
    */
    const model = tf.sequential();
    // 第一层特征提取:
    // 卷积层
    model.add(tf.layers.conv2d({
        // 图片的形状
        inputShape: [28, 28, 1],
        // 卷积核大小, 常设置为奇数, 有中间点
        kernelSize: 5,
        // 特征提取数量
        filters: 8,
        // 移动步长
        strides: 1,
        // 激活函数, relu移除掉不常用的特征
        activation: 'relu',
        // 卷积核初始化方法
        kernelInitializer: 'varianceScaling'
    }));
    // 最大池化层
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    // 第二轮特征提取:
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    // 将高维的filter提取的特征图转化为一维
    model.add(tf.layers.flatten());
    // 全连接层
    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
    }));
    model.compile({
        // 交叉熵损失函数
        loss: 'categoricalCrossentropy',
        // 设置优化器
        optimizer: tf.train.adam(),
        // 设置准确度
        metrics: ['accuracy']
    });
    // 设置训练集和验证集
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(2000);
        return [
            d.xs.reshape([2000, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(200);
        return [
            d.xs.reshape([200, 28, 28, 1]),
            d.labels
        ];
    });

    await model.fit(trainXs, trainYs, {
        validationData: [testXs, testYs],
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    const canvas = document.querySelector('canvas');

    canvas.addEventListener('mousemove', (e) => {
        if (e.buttons === 1) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgb(255,255,255)';
            ctx.fillRect(e.offsetX, e.offsetY, 25, 25);
        }
    });

    window.clear = () => {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.fillRect(0, 0, 300, 300);
    };

    clear();

    window.predict = () => {
        const input = tf.tidy(() => {
            return tf.image.resizeBilinear(
                tf.browser.fromPixels(canvas),
                [28, 28],
                true
            ).slice([0, 0, 0], [28, 28, 1])
            .toFloat()
            .div(255)
            .reshape([1, 28, 28, 1]);
        });
        const pred = model.predict(input).argMax(1);
        alert(`预测结果为 ${pred.dataSync()[0]}`);
    };
};
