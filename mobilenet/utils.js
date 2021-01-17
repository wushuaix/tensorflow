export function file2img(f) {
    return new Promise(resolve => {
        // 从前端界面拿到图片文件, 图片转化为图片的HTNL Element
        const reader = new FileReader();
        reader.readAsDataURL(f);
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.width = 224;
            img.height = 224;
            img.onload = () => resolve(img);
        };
    });
}
