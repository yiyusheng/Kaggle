library(tidyverse)
library(magrittr)
library(jsonlite)
library(caret)
library(lubridate)
library(irlba)
library(xgboost)
set.seed(0)

#---------------------------
cat("Loading data...\n")

tr <- read_csv("~/Data/Kaggle/Google_Analytics_Customer_Revenue_Prediction/train.csv") 
te <- read_csv("~/Data/Kaggle/Google_Analytics_Customer_Revenue_Prediction/test.csv") 

#---------------------------
cat("Defining auxiliary functions...\n")

flatten_json <- . %>% 
  str_c(., collapse = ",") %>% 
  str_c("[", ., "]") %>% 
  fromJSON(flatten = T)

parse <- . %>% 
  bind_cols(flatten_json(.$device)) %>%
  bind_cols(flatten_json(.$geoNetwork)) %>% 
  bind_cols(flatten_json(.$trafficSource)) %>% 
  bind_cols(flatten_json(.$totals)) %>% 
  select(-device, -geoNetwork, -trafficSource, -totals)

is_na_val <- function(x) x %in% c("not available in demo dataset", "(not set)", 
                                  "unknown.unknown", "(not provided)")

has_many_values <- function(x) n_distinct(x) > 1

#---------------------------
cat("Basic preprocessing...\n")

tr <- parse(tr)
te <- parse(te)

y <- log1p(as.numeric(tr$transactionRevenue))
y[is.na(y)] <- 0

tr$transactionRevenue <- NULL
tr$campaignCode <- NULL

id <- te[, "fullVisitorId"]
tri <- 1:nrow(tr)
idx <- ymd(tr$date) < ymd("20170601")

tr_te <- tr %>% 
  bind_rows(te) %>% 
  select_if(has_many_values) %>% 
  mutate_all(funs(ifelse(is_na_val(.), NA, .))) %>% 
  mutate(hits = log1p(as.integer(hits)),
         pageviews = ifelse(is.na(pageviews), 0L, log1p(as.integer(pageviews))),
         visitNumber =  log1p(visitNumber),
         newVisits = ifelse(newVisits == "1", 1L, 0L),
         bounces = ifelse(bounces == "1", 1L, 0L),
         isMobile = ifelse(isMobile, 1L, 0L),
         adwordsClickInfo.isVideoAd = ifelse(!adwordsClickInfo.isVideoAd, 1L, 0L),
         isTrueDirect = ifelse(isTrueDirect, 1L, 0L),
         date = ymd(date),
         year = year(date),
         month = month(date),
         day = day(date),
         wday = wday(date),
         week = week(date),
         yday = yday(date),
         qday = qday(date)) 

#---------------------------
cat("Creating group features...\n")

for (grp in c("month", "day", "wday", "week", "yday", "qday")) {
  col <- paste0(grp, "_user_cnt")
  tr_te %<>% 
    group_by_(grp) %>% 
    mutate(!!col := n_distinct(fullVisitorId)) %>% 
    ungroup()
}

fn <- funs(mean, median, var, min, max, sum, n_distinct, .args = list(na.rm = TRUE))

sum_by_day <- tr_te %>%
  select(day, hits, pageviews) %>% 
  group_by(day) %>% 
  summarise_all(fn) 

sum_by_month <- tr_te %>%
  select(month, hits, pageviews) %>% 
  group_by(month) %>% 
  summarise_all(fn) 

sum_by_dom <- tr_te %>%
  select(networkDomain, hits, pageviews) %>% 
  group_by(networkDomain) %>% 
  summarise_all(fn) 

sum_by_vn <- tr_te %>%
  select(visitNumber, hits, pageviews) %>% 
  group_by(visitNumber) %>% 
  summarise_all(fn) 

sum_by_country <- tr_te %>%
  select(country, hits, pageviews) %>% 
  group_by(country) %>% 
  summarise_all(fn) 

sum_by_city <- tr_te %>%
  select(city, hits, pageviews) %>% 
  group_by(city) %>% 
  summarise_all(fn) 

sum_by_medium <- tr_te %>%
  select(medium, hits, pageviews) %>% 
  group_by(medium) %>% 
  summarise_all(fn) 

sum_by_source <- tr_te %>%
  select(source, hits, pageviews) %>% 
  group_by(source) %>% 
  summarise_all(fn) 

sum_by_ref <- tr_te %>%
  select(referralPath, hits, pageviews) %>% 
  group_by(referralPath) %>% 
  summarise_all(fn) 

#---------------------------
cat("Creating ohe features...\n")

tr_te_ohe <- tr_te %>%
  select(-date, -fullVisitorId, -visitId, -sessionId, visitStartTime) %>% 
  select_if(is.character) %>% 
  mutate_all(factor) %>% 
  mutate_all(fct_lump, prop = 0.025) %>% 
  mutate_all(fct_explicit_na) %>% 
  model.matrix(~.-1, .) %>% 
  as.data.frame() %>% 
  mutate_all(as.integer)

#---------------------------
cat("Joining datasets...\n")

tr_te %<>% 
  select(-date, -fullVisitorId, -visitId, -sessionId, -visitStartTime) %>% 
  left_join(sum_by_city, by = "city", suffix = c("", "_city")) %>% 
  left_join(sum_by_country, by = "country", suffix = c("", "_country")) %>% 
  left_join(sum_by_day, by = "day", suffix = c("", "_day")) %>% 
  left_join(sum_by_dom, by = "networkDomain", suffix = c("", "_dom")) %>% 
  left_join(sum_by_medium, by = "medium", suffix = c("", "medium")) %>% 
  left_join(sum_by_month, by = "month", suffix = c("", "_month")) %>% 
  left_join(sum_by_ref, by = "referralPath", suffix = c("", "_ref")) %>% 
  left_join(sum_by_source, by = "source", suffix = c("", "_source")) %>% 
  left_join(sum_by_vn, by = "visitNumber", suffix = c("", "_vn")) %>% 
  bind_cols(tr_te_ohe) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  select_if(has_many_values) 

rm(tr, te, grp, col, flatten_json, parse, has_many_values, is_na_val,
   fn, sum_by_city, sum_by_country, sum_by_day, sum_by_dom, sum_by_medium, 
   sum_by_month, sum_by_ref, sum_by_source, sum_by_vn, tr_te_ohe)
gc()

#---------------------------
cat("Preparing data...\n")

dtest <- xgb.DMatrix(data = data.matrix(tr_te[-tri, ]))
tr_te <- tr_te[tri, ]
dtr <- xgb.DMatrix(data = data.matrix(tr_te[idx, ]), label = y[idx])
dval <- xgb.DMatrix(data = data.matrix(tr_te[!idx, ]), label = y[!idx])
dtrain <- xgb.DMatrix(data = data.matrix(tr_te), label = y)
cols <- colnames(tr_te)
rm(tr_te, y, tri)
gc()

#---------------------------
cat("Training model...\n")

p <- list(objective = "reg:linear",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 4,
          eta = 0.025,
          max_depth = 8,
          min_child_weight = 5,
          gamma = 0,
          subsample = 0.8,
          colsample_bytree = 0.9,
          alpha = 0,
          lambda = 1)

set.seed(0)
cv <- xgb.train(p, dtr, 5000, list(val = dval), print_every_n = 100, early_stopping_rounds = 250)

nrounds <- round(cv$best_iteration * (1 + sum(!idx) / length(idx)))

set.seed(0)
m_xgb <- xgb.train(p, dtrain, nrounds)

imp <- xgb.importance(cols, model = m_xgb) %T>% 
  xgb.plot.importance(top_n = 25)

#---------------------------
cat("Making predictions...\n")

pred <- predict(m_xgb, dtest) %>% 
  as_tibble() %>% 
  set_names("y") %>% 
  mutate(y = ifelse(y < 0, 0, y)) %>% 
  bind_cols(id) %>% 
  group_by(fullVisitorId) %>% 
  summarise(y = sum(y))

#---------------------------
cat("Making submission file...\n")

read_csv("../input/sample_submission.csv") %>%  
  left_join(pred, by = "fullVisitorId") %>% 
  mutate(PredictedLogRevenue = round(y, 5)) %>% 
  select(-y) %>% 
  write_csv(paste0("tidy_xGb_", round(cv$best_score, 5), ".csv"))
