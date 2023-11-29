# service_predict_car_coast
Project for  FACULTY OF COMPUTER SCIENCE - HSE 

Описание файлов проекта в самом конце...


В рамках данного репозитория реализован проект по созданию сервиса по предсказанию стоимости автомобилей (в рамках домашней работы).
Задача на проект:
  1) Часть 1 - EDA
  2) Часть 2 - Модель только на вещественных признаках
  3) Часть 3 - Добавляем категориальные фичи
  4) Этап fature engineering по генерации новых признаков для моделей линейной регрестии
  5) Финальный этап обучения лениейной модели, замер метрик R2 и MSE (+RMSE)
  6) Создание сервиса с помощью фреймворка FastAPI

1. Часть 1 - EDA
В этом разделе предстояло ответить на ряд вопросов касающихся статистических наблюдений, визуализации данных и обработки признаков.
Были выявлены коррелирующие признаки с целевой переменной, а тае же признаки, которые могли "опасно" корелировать между собой, что могло привести к проблеме мультиколинеарности признаков.
Дана оценка распределению целевой переменной - в ней были незначительное кол-во выбросов

2. Часть 2  Модель только на вещественных признаках
В этой части стояла задача реализации модели только на вещественных признаках. При этом данные были предобрабработаны с помощью StandartScaller.
Реализованы линейные модели предсказания стоимости авто: LinearRegressor, Lasso, а так же Ridge, Elasticnet обученые при помощи подбора гиперпараметров GridSearchCV.
Лучшей оказалась модель LinearRegressor из под коробки, ее метрики:
mse:  226580166397.094
rmse:  476004.3764474167
r2_score:  0.60583000935741

3. Часть 3 - Добавляем категориальные фичи
Так как передомной стояла задача построить линейные модели - необходимо было обработать категориальные переменные, для этого я выбрал OneHotEncoder для создания бинарных столбцов для категориальных переменных.
Для обучения была использована Ridge модель с следующими резуьтатами (результаты, к сожалению, не улучшились)
mse:  226608504274.045
rmse:  476034.1419205612
score:  0.6057807113942733

4. Часть 4 - Feature Engineering
Самая объемная часть, где мною были придуман ряд новых признаков, обработаны пропуски улучшенным методом, добавлена кастомная бизнес-метрика, а так же обученны новые модели "из коробки" и при помощи GridScreachCV.
Лучшей оказалась модель Lasso и снова "из под коробки" победила Lasso + GridScreachCV, ее метрики на тесте значительно превосходят метрики из части 2 и 3.
Данная Lasso модель была взята в качестве "прод решения" и на основе нее реализован сервис в части 5. Метрики модели:
mse:  15220726782.297642
rmse:  123372.30962536788
r2_score:  0.9735212758082409

4. Часть 5 - Реализация сервиса на FastAPI
Сервис был реализован с помощтю фреймворка FastAPI, поднят на сервере Uvicorn, а сам проект написан в PyCharm. В сервисе реализованны 2 ручки - одна для предсказания стоимости одной машины с методом Post и переданным json данными о машине, а вторая ручка предсказывает стоимость для нескольких авто, информация о которых так же передается методом post но в виде CSV файла
Скриншоты сервера будут в конце страницы.

Что удалось:
Самое важное для меня оказалась генерация новых признаков для моделей, особенный "буст" дал метот возведения числовых переменных в квадрат, что стало для меня неожиданностью - сначала я возвел в квадрат только год машины, так как визульно распределение по таргету строилось в виде квадратичной функции, прирост был не большой. Затем я создал все числовые переменные в функцию возведения в квадрат - тем самым максимизировав r2 скор и минимизировав mse скор. Для меня было полезно вспомнить различные методы реализации линейных моделей, поработать с регуляризацией и проведением EDA.
Что не удалось:
К сожалению GridScreachCV ни разу не побил модель из под коробки и оъяснений я не нашел. Была гепотиза, что из за cv=10 (указано в задаии) не хватало данных для валидации модели и какой то из моделей повезло с тестовыми данными и она показала лучший результат, а ее параметры не смогли показать себя с лучшей стороны на всех данных - я изменил на cv=3, но эфекта не добился
Чего не хватает:
Времени! Конечно можно было еще глубже провести EDA, больше признаков создать в Feature Engeneering и разобраться с натройками модели
<img width="1792" alt="POST предсказание стоимости 1 машины" src="https://github.com/Dangennadevich/service_predict_car_coast/assets/86557469/8106f4b8-9a4a-439d-9888-92319e278886">

<img width="1792" alt="POST предсказание стоимости нескольких машин" src="https://github.com/Dangennadevich/service_predict_car_coast/assets/86557469/5c129cd1-0230-45f4-b33a-25f6fc8a2eb9">


