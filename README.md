# Fashion MNIST Classifier

Веб-застосунок для класифікації зображень одягу з використанням нейронних мереж **CNN** та **VGG16**.

Застосунок створений за допомогою **Streamlit**.

## Функціональність

- Вибір між моделями CNN та VGG16
- Завантаження зображення користувачем
- Відображення завантаженого зображення
- Класифікація зображення
- Відображення передбаченого класу
- Відображення результатів для всіх класів
- Візуалізація Accuracy та Loss моделей

## Структура проєкту

```text
Home_work/
├── app.py
├── README.md
├── models/
│   ├── cnn_model.keras
│   └── vgg16_model.keras
└── data/
    ├── cnn_history.csv
    └── vgg16_history.csv
```

## Використані бібліотеки

- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Pillow

## Запуск

Встановити необхідні бібліотеки:

```bash
pip install tensorflow streamlit pandas numpy matplotlib pillow
```

Запустити застосунок:

```bash
streamlit run app.py
```

Після запуску необхідно вибрати модель, завантажити зображення та натиснути кнопку **«Класифікувати»**.
