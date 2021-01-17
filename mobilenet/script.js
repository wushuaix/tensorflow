/*
* 使用事先训练好的模型进行图片分类
* 已经事先训练好的模型, 无需训练即可预测
* 在Tensorflow.js中可以调用Web格式的模型文件
*
* */

import * as tf from '@tensorflow/tfjs';
import { IMAGENET_CLASSES } from './imagenet_classes';
import { file2img } from './utils';

// 训练好的模型地址
const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json';

window.onload = async () => {
    // loadLayersModel方法加载MobileNet模型(一种卷积神经元网络模型, 体积小, 相应速度快)
    const model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    // 进行预测
    window.predict = async (file) => {
        const img = await file2img(file);
        document.body.appendChild(img);
        const pred = tf.tidy(() => { // tidy方法清除缓存,防止内存泄漏
            // 将图片的HTNL Element转化为Tensor, Tensor归一化模型需要的格式
            const input = tf.browser.fromPixels(img) // 先拿到图片版的Tensor
                .toFloat() // 整数格式转化为浮点数格式
                .sub(255 / 2) // 归一化处理到[-1, 1]之间
                .div(255 / 2)
                .reshape([1, 224, 224, 3]); // 预测一个224*224的彩色图片
            // 调用predict进行预测
            return model.predict(input);
        });

        const index = pred.argMax(1).dataSync()[0];
        // 弹框预测结果
        setTimeout(() => {
            alert(`预测结果：${IMAGENET_CLASSES[index]}`);
        }, 0);
    };
};
