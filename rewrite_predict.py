import nbformat as nbf
from pathlib import Path


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Plant Disease Detection - Prediction Notebook\n\n"
            "Bu notebook, PlantVillage ve PlantDoc için kaydedilmiş lojistik regresyon "
            "modellerini yükleyip tek bir test görseli üzerinde tahmin yapar."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import numpy as np\n"
            "from PIL import Image\n"
            "import os\n"
            "import math\n"
            "import random"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Logistic Regression sınıfı (load ile uyumlu)\n\n"
            "Not: Buradaki load metodu, training tarafında save() ile kaydettiğin .npz "
            "dosyasıyla uyumlu."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "def softmax(logits):\n"
            "    max_logit = max(logits)\n"
            "    exps = [math.exp(z - max_logit) for z in logits]\n"
            "    s = sum(exps)\n"
            "    if s == 0.0:\n"
            "        c = len(logits)\n"
            "        return [1.0 / c for _ in range(c)]\n"
            "    return [e / s for e in exps]\n"
            "\n"
            "\n"
            "class MulticlassLogisticRegression:\n"
            "    def __init__(self, num_features, num_classes, learning_rate=0.1):\n"
            "        self.num_features = num_features\n"
            "        self.num_classes = num_classes\n"
            "        self.learning_rate = learning_rate\n"
            "\n"
            "        # dummy init, real weights will be loaded from file\n"
            "        self.W = [\n"
            "            [(random.random() - 0.5) * 0.01 for _ in range(num_classes)]\n"
            "            for _ in range(num_features)\n"
            "        ]\n"
            "        self.b = [(random.random() - 0.5) * 0.01 for _ in range(num_classes)]\n"
            "\n"
            "    def _compute_logits(self, x):\n"
            "        logits = [0.0 for _ in range(self.num_classes)]\n"
            "        for k in range(self.num_classes):\n"
            "            s = 0.0\n"
            "            for j in range(self.num_features):\n"
            "                s += x[j] * self.W[j][k]\n"
            "            s += self.b[k]\n"
            "            logits[k] = s\n"
            "        return logits\n"
            "\n"
            "    def predict_proba_one(self, x):\n"
            "        logits = self._compute_logits(x)\n"
            "        probs = softmax(logits)\n"
            "        return probs\n"
            "\n"
            "    def predict_one(self, x):\n"
            "        probs = self.predict_proba_one(x)\n"
            "        best_class = 0\n"
            "        best_prob = probs[0]\n"
            "        for k in range(1, self.num_classes):\n"
            "            if probs[k] > best_prob:\n"
            "                best_prob = probs[k]\n"
            "                best_class = k\n"
            "        return best_class\n"
            "\n"
            "    def predict(self, X):\n"
            "        return [self.predict_one(x) for x in X]\n"
            "\n"
            "    @classmethod\n"
            "    def load(cls, path):\n"
            "        data = np.load(path)\n"
            "\n"
            "        num_features = int(data['num_features'])\n"
            "        num_classes = int(data['num_classes'])\n"
            "\n"
            "        model = cls(\n"
            "            num_features=num_features,\n"
            "            num_classes=num_classes,\n"
            "            learning_rate=0.01\n"
            "        )\n"
            "\n"
            "        W_array = data['W']\n"
            "        b_array = data['b']\n"
            "\n"
            "        model.W = W_array.tolist()\n"
            "        model.b = b_array.tolist()\n"
            "\n"
            "        print(f\"Model loaded from {path}\")\n"
            "        print('num_features:', num_features)\n"
            "        print('num_classes:', num_classes)\n"
            "\n"
            "        return model\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## Görüntüyü 32×32 gri vektöre çevirme")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "IMG_SIZE = 32\n"
            "\n"
            "def load_image_as_vector(path, img_size=IMG_SIZE):\n"
            "    with Image.open(path) as img:\n"
            "        img = img.convert('L')\n"
            "        img = img.resize((img_size, img_size))\n"
            "        pixels = list(img.getdata())\n"
            "        vector = [p / 255.0 for p in pixels]\n"
            "        return vector\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## PlantVillage sınıf isimlerini klasörden oku\n\n"
            'predict.ipynb, "Soruce Code" klasöründe ise kök klasörü şu şekilde '
            "tanımlıyoruz. Bu, PlantVillage için tüm sınıfları (klasör isimlerini) "
            "listeleyecek."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "def get_class_names(root_dir):\n"
            "    names = []\n"
            "    for item in os.listdir(root_dir):\n"
            "        full_path = os.path.join(root_dir, item)\n"
            "        if os.path.isdir(full_path):\n"
            "            names.append(item)\n"
            "    names = sorted(names)\n"
            "    return names\n"
            "\n"
            "# predict.ipynb, \"Soruce Code\" klasöründe ise:\n"
            "PLANTVILLAGE_ROOT = \"../Datasets/plantvillage/color\"\n"
            "\n"
            "class_names_pv = get_class_names(PLANTVILLAGE_ROOT)\n"
            "idx_to_class_pv = {idx: name for idx, name in enumerate(class_names_pv)}\n"
            "\n"
            "print(\"PlantVillage classes:\")\n"
            "for idx, name in enumerate(class_names_pv):\n"
            "    print(f\"{idx:2d} -> {name}\")\n"
            "\n"
            "# Bu, PlantVillage için tüm sınıfları (klasör isimlerini) listeleyecek ✅\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## PlantDoc için sınıf listesi\n\n"
            "Henüz net isim listemiz yoksa şimdilik generic isim üretelim. İleride "
            "istersen plantdoc_classes.txt gibi bir dosyada gerçek isimleri tutup "
            "buradan okuyabilirsin. Şimdilik generic (ama tüm sınıfları yazdıran) "
            "versiyon. Bu kısmı bir sonraki hücrede tamamlayacağız; önce modelleri "
            "yükleyelim."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Modelleri dosyadan yükle\n\n"
            "Models are saved under \"models\" folder next to this notebook. "
            "Bu hücreyi mutlaka çalıştır. Aksi halde NameError alırsın."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# Models are saved under \"models\" folder next to this notebook\n"
            "MODEL_PV_PATH = \"models/logreg_plantvillage.npz\"\n"
            "MODEL_PD_PATH = \"models/logreg_plantdoc.npz\"\n"
            "\n"
            "model_pv = MulticlassLogisticRegression.load(MODEL_PV_PATH)\n"
            "model_pd = MulticlassLogisticRegression.load(MODEL_PD_PATH)\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## PlantDoc sınıf isimlerini üret ve yazdır\n\n"
            "Model yüklendikten sonra model_pd.num_classes üzerinden sınıf sayısını "
            "biliyoruz. Şimdilik isimleri \"PlantDoc_class_0, PlantDoc_class_1, …\" "
            "gibi yapalım. Eğer ileride gerçek isimleri (Apple_scab, Tomato_blight vs.) "
            "yazmak istersen plantdoc_classes.txt diye bir dosya yapıp (her satıra bir "
            "sınıf adı) bu listeden class_names_pd'yi doldurabilirsin."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "num_classes_pd = model_pd.num_classes\n"
            "\n"
            "# For now, create generic class names\n"
            "class_names_pd = [f\"PlantDoc_class_{i}\" for i in range(num_classes_pd)]\n"
            "idx_to_class_pd = {idx: name for idx, name in enumerate(class_names_pd)}\n"
            "\n"
            "print(\"PlantDoc classes (generic names):\")\n"
            "for idx, name in enumerate(class_names_pd):\n"
            "    print(f\"{idx:2d} -> {name}\")\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## Tahmin fonksiyonları (iki model için)")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "def predict_with_plantvillage(image_path):\n"
            "    \"\"\"\n"
            "    Predict class for a given image using PlantVillage model (model_pv).\n"
            "    Returns predicted index and class name.\n"
            "    \"\"\"\n"
            "    x_vec = load_image_as_vector(image_path, img_size=IMG_SIZE)\n"
            "    pred_idx = model_pv.predict_one(x_vec)\n"
            "    class_name = idx_to_class_pv.get(pred_idx, f\"unknown_{pred_idx}\")\n"
            "    print(f\"[PlantVillage] predicted class index = {pred_idx}\")\n"
            "    print(f\"[PlantVillage] predicted class name  = {class_name}\")\n"
            "    return pred_idx, class_name\n"
            "\n"
            "\n"
            "def predict_with_plantdoc(image_path):\n"
            "    \"\"\"\n"
            "    Predict class for a given image using PlantDoc model (model_pd).\n"
            "    Returns predicted index and generic class name.\n"
            "    \"\"\"\n"
            "    x_vec = load_image_as_vector(image_path, img_size=IMG_SIZE)\n"
            "    pred_idx = model_pd.predict_one(x_vec)\n"
            "    class_name = idx_to_class_pd.get(pred_idx, f\"unknown_{pred_idx}\")\n"
            "    print(f\"[PlantDoc] predicted class index = {pred_idx}\")\n"
            "    print(f\"[PlantDoc] predicted class name  = {class_name}\")\n"
            "    return pred_idx, class_name\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Gerçek test\n\n"
            "Bu şekilde hem PlantVillage hem PlantDoc modeli dosyadan yüklenecek, tüm "
            "sınıflar bir kez konsola yazılacak ve seçtiğin test görseli için iki "
            "modelin tahmini de görünecek."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "test_image_path = \"../TestImages/apple_healty.jpg\"  # kendi path'ini gir\n"
            "\n"
            "print(\"Testing with PlantVillage model:\")\n"
            "pv_idx, pv_name = predict_with_plantvillage(test_image_path)\n"
            "\n"
            "print(\"\\nTesting with PlantDoc model:\")\n"
            "pd_idx, pd_name = predict_with_plantdoc(test_image_path)\n"
        )
    )

    nb["cells"] = cells
    return nb


def main():
    nb = build_notebook()
    out_path = Path("predict.ipynb")
    out_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Notebook rewritten: {out_path}")


if __name__ == "__main__":
    main()

