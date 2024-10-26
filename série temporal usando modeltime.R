################################################################################
#          SCRIPT DO ESTUDO DE SÉRIES TEMPORAIS USANDO A BIBLIOTECA            #
#          MODELTIME PARA USO DE MODELOS DE MACHINE LEARNING EM SÉRIES         #
#          TEMPORAIS E USO DE MODELOS HÍBRIDOS - NOVAS VISUALIZAÇÕES           #
#          ENTRE OUTRAS COISAS, UTILIZAÇÃO DE SÉRIES ECONÔMICAS VIA            #
#          PACOTE BETS, COM O OBJETIVO DE TREINAR OS MELHORES MODELOS          #
################################################################################
rm(list = ls(all = T))
gc()
################################################################################
#                           CARREGANDO AS BIBLIOTECAS                          #
################################################################################
library(dplyr)
library(lubridate)
library(modeltime)
library(rsample)
library(parsnip)
library(tidymodels)
library(tidyverse)
library(timetk)
library(forecast)
library(zoo)
library(modeltime.resample)
library(modeltime.ensemble)
library(BETS)
library(ipeadatar)
library(DT)
################################################################################
#               COMEÇANDO A ESTRUTURAR A MINHA SÉRIE TEMPORAL                  #
################################################################################
series <- ipeadatar::search_series()
serie <- ipeadata("DERAL12_ATSCAR12", language = "br") %>% 
  select(date, value)
################################################################################
#       A SÉRIE TEMPORAL QUE EU ESCOLHI PARA TRABALHAR ELA É DO IPEADATA       #
#       E REFERE-SE A CONCESSÕES DE CRÉDITO TOTAL - SÉRIE MENSAL               #
################################################################################
min(serie$date); max(serie$date)
skimr::skim(serie)            # UMA BREVE ESTATÍSTICA DESCRITIVA DA MINHA SÉRIE 
################################################################################
#                           VENDO A VISUALIZAÇÃO DA SÉRIE                      #
################################################################################

interativo <- TRUE
serie %>% 
  plot_time_series(date, value, .interactive = interativo, .title = "Concessões de Crédito", 
                   .plotly_slider = TRUE) 
################################################################################
#                               CRIANDO A RECEITA                              #
################################################################################
ts_data <- ts(serie$value, frequency = 12)

evaluate_k <- function(K) {
  fourier_terms <- fourier(ts_data, K = K)
  model <- auto.arima(ts_data, xreg = fourier_terms)
  AIC(model)  # Ou use BIC(model)
}

# Testar valores de K de 1 a 5
results <- sapply(1:5, evaluate_k)
optimal_k <- which.min(results)
print(optimal_k)


serie <- serie %>% 
  mutate(month = factor(month(date, label = TRUE), ordered = FALSE))

rec_feat_engine <- recipe(value ~ ., data = serie) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_fourier(date, period = 365, K = optimal_k)

serie <- rec_feat_engine %>% 
  prep() %>% 
  bake(new_data = NULL)
fmla <- as.formula(paste0("value ~ date + ", paste0(names(serie)[-c(1,2)], collapse = " + ")))
################################################################################
#               REALIZANDO O SPLIT PARA TREINAR E TESTAR OS MODELOS            #
################################################################################
splits <- serie %>% 
  time_series_split(assess = "12 months", cumulative = TRUE)

splits %>% 
  tk_time_series_cv_plan () %>% 
  plot_time_series_cv_plan(date, value, 
    .interactive =  interativo,
    .title = "Gráfico de Teste e Treinamento do modelo", 
    .y_lab = "Índice", 
    .x_lab = "Tempo", 
    .plotly_slider = TRUE
  )
################################################################################
#                             DEFININDO OS MODELOS                             #
################################################################################

# Modelo Auto-Arima 

model_fit_auto_arima <- arima_reg() %>% 
  set_engine(engine = "auto_arima") %>% 
  fit(value ~ date, training(splits))

# Modelo Arima - Boost

model_arima_boost <- arima_boost(
  min_n = 2, 
  learn_rate = 0.010
) %>% 
  set_engine(engine = "auto_arima_xgboost") %>% 
  fit(value ~ date, data = training(splits))

# Modelo ES - Suavização Exponencial 

model_fit_es <- exp_smoothing() %>% 
  set_engine(engine = "ets") %>% 
  fit(value ~ date, data = training(splits))

# Modelo Prophet - Prophet

model_prophet <- prophet_reg() %>% 
  set_engine(engine = "prophet") %>% 
  fit(value ~ date, data = training(splits))

# Modelo Prophet - Com variáveis exógenas

model_prophet_exo <- prophet_reg(
  changepoint_num = 50
) %>% 
  set_engine("prophet") %>% 
  fit(fmla, training(splits))


# Modelo de Regressão Linear - LM

model_fit_lm <- linear_reg() %>% 
  set_engine("lm") %>% 
  fit(value ~ date, data = training(splits))

# Modelo Adaptado de Regressão Multivariada

model_spec_mars <- mars(mode = "regression") %>% 
  set_engine("earth")

recipe_spec <- recipe(value ~ date, data = training(splits)) %>% 
  step_date(date, features = "month", ordinal = F) %>% 
  step_mutate(date_num = as.numeric(date)) %>% 
  step_normalize(date_num) %>% 
  step_rm(date)

wflw_fit_mars <- workflow() %>% 
  add_recipe(recipe_spec) %>% 
  add_model(model_spec_mars) %>% 
  fit(training(splits))

# Modelo de Rede Neural 

model_rede_neural <- nnetar_reg(mode = "regression") %>% 
  set_engine(engine = "nnetar") %>% 
  fit(value ~ date, data = training(splits))

# Modelo XGBOOST 

model_xgboost <- boost_tree(mode = "regression") %>% 
  set_engine("xgboost") %>% 
  fit(fmla, data = training(splits))

# Modelo TABTS - com variáveis exógenas

model_stml_tbats_exo <- seasonal_reg(mode = "regression") %>% 
  set_engine("tbats") %>% 
  fit(fmla, data = training(splits))


# Modelo TABTS 

model_stml_tbats <- seasonal_reg(mode = "regression") %>% 
  set_engine("tbats") %>% 
  fit(value ~ date, data = training(splits))

# Modelo Perceptron Multicamadas com 5 camadas ocultas 

model_nnetar5 <- nnetar_reg(
  hidden_units = 5, 
  num_networks = 30
) %>% 
  set_engine("nnetar") %>% 
  fit(value ~ date, training(splits))

# Modelo perceptron Multicamadas com 10 camadas ocultas 

model_nnetar10 <- nnetar_reg(
  hidden_units = 10, 
  num_networks = 30
) %>% 
  set_engine("nnetar") %>% 
  fit(value ~ date, training(splits))

# Modelo perceptron Multicamadas com 20 camadas ocultas

model_nnetar20 <- nnetar_reg(
  hidden_units = 20, 
  num_networks = 30
) %>% 
  set_engine("nnetar") %>% 
  fit(fmla, training(splits))

# Modelo perceptron Multicamadas com 30 camadas ocultas 

model_nnetar30 <- nnetar_reg(
  hidden_units = 30, 
  num_networks = 30
) %>% 
  set_engine("nnetar") %>% 
  fit(value ~ date, training(splits))

# Modelo de Floresta Aleatória - com variáveis exógenas

model_random_forest_exo <- rand_forest() %>%
  set_mode("regression") %>% 
  set_engine("ranger") %>% 
  fit(fmla, training(splits))

# Modelo Random Forest - sem variáveis exógenas

model_random_forest <- rand_forest() %>% 
  set_mode("regression") %>% 
  set_engine("ranger") %>% 
  fit(value ~ date, training(splits))

# Modelo SVM - com variáveis exógenas

model_svm_exo <- svm_rbf() %>% 
  set_mode("regression") %>% 
  set_engine("kernlab") %>% 
  fit(fmla, training(splits))

# Modelo SVM 

model_svm <- svm_rbf() %>% 
  set_mode("regression") %>% 
  set_engine("kernlab") %>% 
  fit(value ~ date, training(splits))

# Modelo SVM - Polinomial

model_svm_poli <- svm_poly() %>% 
  set_mode("regression") %>% 
  set_engine("kernlab") %>% 
  fit(value~date, training(splits))

# Modelo ADAM - com variáveis exógenas

model_adam_exo <- adam_reg() %>% 
  set_mode("regression") %>% 
  set_engine("adam") %>% 
  fit(fmla, training(splits))

# Modelo ADAM 

model_adam <- adam_reg() %>% 
  set_mode("regression") %>% 
  set_engine("adam") %>% 
  fit(value ~ date, training(splits))

################################################################################
#                     TABELA COM TODOS OS MODELOS                              #
#   OBSERVAÇÃO: MODELOS DE REGRESSÃO USANDO APENAS UMA VARIÁVEL REGRESSORA     #
#        QUE NESSE CASO É A DATA, A DATA PARA PREVER O VALOR                   #
################################################################################

model_tbl <- modeltime_table(
  model_adam, 
  model_fit_auto_arima, 
  model_arima_boost, 
  model_fit_es, 
  model_fit_lm, 
  model_nnetar5, 
  model_nnetar10, 
  model_nnetar20, 
  model_nnetar30, 
  model_prophet, 
  model_random_forest, 
  model_rede_neural, 
  wflw_fit_mars, 
  model_stml_tbats, 
  model_xgboost, 
  model_svm, 
  model_prophet_exo, 
  model_adam_exo, 
  model_random_forest_exo, 
  model_stml_tbats_exo, 
  model_svm_poli, 
  model_svm_exo
) %>% 
  update_modeltime_description(rep(1:22), c("ADAM", 
                                            "AUTO - ARIMA", 
                                            "ARIMA XGBOOST", 
                                            "SUAVIZAÇÃO EXPONENCIAL", 
                                            "REGRESSÃO LINEAR",
                                            "PERCEPTRON (5 CAMADAS)", 
                                            "PERCEPTRON (10 CAMADAS)", 
                                            "PERCEPTRON (15 CAMADAS)", 
                                            "PERCEPTRON (20 CAMADAS)", 
                                            "PROPHET", 
                                            "RANDOM FOREST", 
                                            "REDE NEURAL", 
                                            "MARS", 
                                            "TBATS", 
                                            "XGBOOST", 
                                            "SVM", 
                                            "PROPHET (VAR. EXÓGENAS)", 
                                            "ADAM (VAR. EXÓGENAS)", 
                                            "RANDOM FOREST (VAR. EXÓGENAS)", 
                                            "TBATS (VAR. EXÓGENAS)", 
                                            "SVM POLINOMIAL", 
                                            "SVM (VAR. EXÓGENAS)"
                                            ))
model_tbl

################################################################################
#                                   CALIBRANDO                                 #
################################################################################

calibre_tbl <- model_tbl %>% 
  modeltime_calibrate(testing(splits))

################################################################################
#                           AVALIANDO OS MELHORES MODELOS                      #
################################################################################

meas <- calibre_tbl %>% 
  modeltime_accuracy(metric_set = metric_set(mae, rmse,rsq))

################################################################################
#                     OBSERVANDO OS 10 MELHORES MODELOS                        #
################################################################################

meas_ordenado <- meas %>% 
  arrange(mae, rmse, rsq)

meas_ordenado
################################################################################
#                       VISUALIZANDO OS RESULTADOS                             #
################################################################################

calibre_tbl %>% 
  modeltime_forecast(
    new_data = testing(splits), 
    actual_data = serie, 
    conf_interval = 0) %>% 
  plot_modeltime_forecast(
    .interactive = interativo, 
    .title = "Gráfico do Ajuste dos modelos", 
    .color_lab = "Legenda", 
    .plotly_slider = TRUE)


resultado <- calibre_tbl %>% 
  filter(.model_desc %in% meas_ordenado$.model_desc[1:5]) %>% 
  mutate(order = match(.model_desc, meas_ordenado$.model_desc)) %>% 
  arrange(order) %>% 
  select(.calibration_data, .model_desc)


banco_selecionado_teste <- bind_rows(resultado) %>% 
  unnest(cols = .calibration_data) %>% 
  rename_with(~gsub("\\.", "_", .), everything())

banco_selecionado_teste
################################################################################
#                     COMBINAÇÃO DOS 10 MELHORES MODELOS                       #
################################################################################

best_model_tbl <- model_tbl %>% 
  filter(.model_id %in% meas_ordenado$.model_id[1:10])

ensemble_fit_media <- best_model_tbl %>%  # Modelo Combinado pela média  
  ensemble_average(type = "mean")

ensemble_fit_media

ensemble_fit_mediana <- best_model_tbl %>% # Modelo Combinado pela Mediana 
  ensemble_average(type = "median")

ensemble_fit_mediana

w <- 1/meas_ordenado$mae[1:10]
ensemble_fit_weighted_media <- best_model_tbl %>% 
  ensemble_weighted(loadings = w)

ensemble_fit_weighted_media

################################################################################
#                       TABELA DOS MODELOS COMBINADOS                          #
################################################################################

ens_calib_tbl <- modeltime_table(
  ensemble_fit_media, 
  ensemble_fit_mediana, 
  ensemble_fit_weighted_media
) %>% 
  modeltime_calibrate(testing(splits)) %>% 
  update_modeltime_description(rep(1:3), c("Hibridização por Média", 
                                           "Hibridização por Mediana", 
                                           "Hibridização por Pesos"))

################################################################################
#                           MÉTRICAS DE AVALIAÇÃO                              #
################################################################################

meas_model_hibrido <- ens_calib_tbl %>% 
  modeltime_accuracy(metric_set = metric_set(mae, rmse))

meas_model_hibrido

meas_model_hibrido_ordenado <- meas_model_hibrido %>%  
  arrange(mae, rmse)

meas_model_hibrido_ordenado

################################################################################
#              VISUALIZANDO OS GRÁFICOS DOS MODELOS COMBINADOS                 #
################################################################################

ens_calib_tbl %>% 
  modeltime_forecast(
    new_data = testing(splits), 
    actual_data = serie, 
    conf_interval = 0) %>% 
  plot_modeltime_forecast(
    .interactive = interativo, 
    .title = "Gráfico do Ajuste dos modelos Híbridos", 
    .color_lab = "Legenda", 
    .plotly_slider = TRUE)

modelos_hibridos_resultados <- ens_calib_tbl %>% 
  filter(.model_desc == meas_model_hibrido_ordenado$.model_desc[1]) %>% 
  select(.calibration_data, .model_desc)

modelos_hibridos_resultados

modelos_hibridos_resultados_banco <- modelos_hibridos_resultados[[1]][[1]] %>% 
  mutate(modelo = modelos_hibridos_resultados[[2]][[1]])

names(modelos_hibridos_resultados_banco) <- c("date","_actual","_prediction","_residuals","_model_desc")

banco_teste_resultados <- bind_rows(banco_selecionado_teste, modelos_hibridos_resultados_banco)

################################################################################
#         CRIANDO UMA TABELA PARA MOSTRAR OS MELHORES MODELOS DE ACORDO COM    #
#                     OS MELHORES MODELOS NO BANCO DE TESTE                    #
################################################################################
banco_teste_resultados %>% 
  mutate(data = as.yearmon(date), 
         data = as.character(data)) %>%
  select(data, `_model_desc`,`_actual`, `_prediction`, `_residuals`) %>% 
datatable(extensions = c("Buttons", "FixedHeader"), 
          filter = "top", 
          caption = "Tabela de Avaliação dos Melhores Modelos",
          rownames = FALSE,
          options = list(
    dom = "Bfrtip", 
    pageLength = 12, 
    autoWidth = TRUE,
    fixedHeader = TRUE,
    buttons = c("copy", "csv", "excel", "pdf", "print"), 
    initComplete = JS(
      "function(settings, json) {",
      "$(this.api().table().header()).css({'background-color': '#808080', 'color': '#fff'});",
      "}"), 
    columnDefs = list(
      list(className = 'dt-center', targets = '_all')
  )), colnames = c("Data", "Modelo", "Valor Real", "Valor Predito", "Resíduos")
)%>%
  formatStyle(
    '_residuals',
    background = styleColorBar(abs(banco_teste_resultados$`_residuals`), 'steelblue'),
    backgroundSize = '100% 90%',
    backgroundRepeat = 'no-repeat',
    backgroundPosition = 'center'
  ) %>% 
  formatCurrency(c("_actual", "_prediction", "_residuals"), currency = "", interval = 3, mark = ".", dec.mark = ",")

################################################################################
#                  VISUALIZANDO SOMENTE OS MELHORES MODELOS                    #
################################################################################

modelos_melhores_tables <- bind_rows(calibre_tbl, ens_calib_tbl)

modelos_melhores_tables <- modelos_melhores_tables %>% 
  filter(.model_desc %in% meas_ordenado$.model_desc[1:5]|.model_desc == meas_model_hibrido_ordenado$.model_desc[1])

modelos_melhores_tables %>% 
  modeltime_forecast(
    new_data = testing(splits), 
    actual_data = serie, 
    conf_interval = 0.80) %>% 
  plot_modeltime_forecast(
    .interactive = interativo, 
    .title = "Melhores Modelos", 
    .color_lab = "Legenda", 
    .plotly_slider = TRUE)

################################################################################
#                          REFITAR E PREVER NOVOS VALORES                      #
################################################################################

refit_tabela <- modelos_melhores_tables %>% 
  modeltime_refit(data = serie)

################################################################################
#                             PREVISÃO DE NOVOS VALORES                        #
################################################################################

refit_tabela %>% 
  modeltime_forecast(
    new_data = serie,
    h = "1 year", 
    actual_data = serie, 
    conf_interval = 0
  ) %>% 
  plot_modeltime_forecast(
    .interactive = interativo, 
    .title = "Modelo Retreinado", 
    .color_lab = "Legenda", 
    .plotly_slider = TRUE
  )

novas_previsões <- refit_tabela %>% 
  modeltime_forecast(
    new_data = serie, 
    h = "1 year", 
    actual_data = serie)

novas_previsões <- novas_previsões %>% 
  filter(.model_desc != "ACTUAL")
################################################################################
#         CRIANDO UMA TABELA PARA MOSTRAR OS MELHORES MODELOS DE ACORDO COM    #
#                     OS MELHORES MODELOS NO BANCO REFITADO                    #
################################################################################
novas_previsões %>% 
  mutate(data = as.yearmon(.index), 
         data = as.character(data)) %>%
  select(data, .model_desc, .value, .conf_lo, .conf_hi) %>% 
  datatable(extensions = c("Buttons", "FixedHeader"), 
            filter = "top", 
            caption = "Tabela de Previsão dos Novos Valores",
            rownames = FALSE,
            options = list(
              dom = "Bfrtip", 
              pageLength = 12, 
              autoWidth = TRUE,
              fixedHeader = TRUE,
              buttons = c("copy", "csv", "excel", "pdf", "print"), 
              initComplete = JS(
                "function(settings, json) {",
                "$(this.api().table().header()).css({'background-color': '#808080', 'color': '#fff'});",
                  "}"), 
              columnDefs = list(
                list(className = 'dt-center', targets = '_all')
              )), colnames = c("Data", "Modelo", "Valor Pontual", "Limite Inferior", "Limite Superior")
    ) %>% 
  formatCurrency(c(".value", ".conf_lo", ".conf_hi"), currency = "", interval = 3, mark = ".", dec.mark = ",")
  