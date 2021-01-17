// 迁移学习: 把已经训练好的模型参数迁移到新的模型来帮助新模型训练
// 深度学习模型参数多, 从头训练成本高
// 删除原始模型的最后一层, 基于此截断模型的输出训练一个新的(通常相当浅的)模型
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getInputs } from './data';
import { img2x, file2img } from './utils';

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json';
const NUM_CLASSES = 3;
const BRAND_CLASSES = ['android', 'apple', 'windows'];

window.onload = async () => {
    const { inputs, labels } = await getInputs();
    const surface = tfvis.visor().surface({ name: '输入示例', styles: { height: 250 } });
    inputs.forEach(img => {
        surface.drawArea.appendChild(img);
    });

    // 定义模型结构: 截断模型 + 双层神经网络
    // 加载MobileNet模型并截断
    // 截断模型作为输入, 双层神经网络作为输出
    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    // 模型概况显示summary
    mobilenet.summary();
    // 截断模型
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    const truncatedMobilenet = tf.model({
        inputs: mobilenet.inputs, // 输入为mobilenet的输入
        outputs: layer.output // 输出为截断模型
    });

    const model = tf.sequential();
    // 将高维的filter提取的特征图转化为一维, 无训练操作
    model.add(tf.layers.flatten({
        inputShape: layer.outputShape.slice(1)
    }));
    // 双层神经网络构造
    model.add(tf.layers.dense({
        units: 10,
        activation: 'relu'
    }));
    // 全连接层做分类
    model.add(tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax'
    }));
    // 设置交叉熵损失函数, 设置优化器和准确度进行训练
    model.compile({ loss: 'categoricalCrossentropy', optimizer: tf.train.adam() });
    // 将训练数据输入到截断模型中, 截断模型输出的数据为下次的输入数据
    const { xs, ys } = tf.tidy(() => {
        const xs = tf.concat(inputs.map(imgEl => truncatedMobilenet.predict(img2x(imgEl))));
        const ys = tf.tensor(labels);
        return { xs, ys };
    });
    // 模型训练
    await model.fit(xs, ys, {
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    // 进行预测
    window.predict = async (file) => {
        const img = await file2img(file);
        document.body.appendChild(img);
        const pred = tf.tidy(() => {
            const x = img2x(img);
            const input = truncatedMobilenet.predict(x);
            return model.predict(input);
        });

        const index = pred.argMax(1).dataSync()[0];
        setTimeout(() => {
            alert(`预测结果：${BRAND_CLASSES[index]}`);
        }, 0);
    };

    window.download = async () => {
        await model.save('downloads://model');
    };
};
