#
# author: L. Beuchée, T. Guyet
# date: 2/2022
#

library(tidyverse)
library(tsibble)
library(e1071)
library(lubridate)
library(forecast)
library(stringr)
library(randomForest)

#Configuration des variables d'environnement utiles
rm(list=ls())

h=93 # horizon de prédiction
k=50 # nombre de valeurs à utiliser dans le passer
max_NA_number = 50 # max number of NA in a time series to handle it

args <- commandArgs( trailingOnly = TRUE )
if( length(args) == 0 ) {
  rep_data = "../data_collection/"
  #rep_data = "/home/tguyet/Stages/2021/Lola/projetpiezo/data2"
  rep_results = "./"
  datasetfile = 'dataset_2015_2021.csv'
  prefix = ''
} else {
  rep_data = dirname(args[1])
  rep_results = rep_data
  datasetfile = basename(args[1])
  prefix = paste0(substring(datasetfile,1,nchar(datasetfile)-4),"_")
}

#chargement des données
data <- read_delim(paste0(rep_data,"/",datasetfile), ",")
code_bss <-data$bss %>% unique()

data <- distinct(data, bss, time, .keep_all= TRUE)

# il y a plusieurs données manquantes pour les données de niveau d'eau, on fait donc une imputation de données manquantes

#D'abord, on se débarrasse des séries pour lesquelles il manque trop de données
# -> moins de max_NA_number (mais au moins 1)
code_bss=(data %>% filter( is.na(p)) %>% group_by( bss) %>% summarize( n=n()) %>% filter(n<max_NA_number))$bss
# -> plus, ceux avec aucun NA
code_bss=union(code_bss, (data%>%group_by( bss) %>% summarize( n=sum(p) ) %>% filter( !is.na(n) ))$bss )
data = data %>% filter( bss %in% code_bss )

# On applique une imputation sur chaque sous-groupe
ret = data %>% group_by( bss ) %>% group_map(
  ~ as.vector(na.interp(.x$p))
)
data$p = unlist(ret)

all_rmsse <- NULL
for ( bss_id in 1:length(code_bss) ){
  cat(paste0("processing time series: ",bss_id,"/",length(code_bss)," [at ",format(Sys.time(),"%H:%M"),"]\n"))
  data_bss <-  data %>% filter(bss ==code_bss[bss_id]) %>% select(time,p,tp,e,bss)
  
  #compute time series approximation on the train part
  approx <- data_bss %>% group_by( yday(time) ) %>% summarize( tp = mean(tp), e = mean(e) )
  #reconstruction from oct, 15th (288th day of the year), of length h
  if(h>78){
    approx <- bind_rows(approx[-(1:288),],approx[(1:(h-366+288)),])
  }else{
    approx <- approx[288:(288+h),]
  }
  
  train_df <-  data_bss %>%filter( time<"2020-10-15" ) %>% select(p,tp,e)
  n <- nrow(train_df)
  #on creer la matrice qui nous permettras d'appliquer un modèle de prédiction
  partial <- list()
  partial[[1]] <- (data_bss%>%filter( time<"2020-10-15" ))[-(1:k),]
  for (j in 1:(k-1)){
    partial[[j+1]] = train_df[-c(1:(k-j),(n-j+1):n),]
  } 
  partial[[k+1]] = train_df[-((n-k+1):n),]
  train = bind_cols(partial, .name_repair = ~ vctrs::vec_as_names(..., repair = "unique", quiet = TRUE))
  rm(partial)
  train = train %>% relocate( starts_with("p"), .after=last_col()) %>% 
    relocate( starts_with("tp"), .after=last_col()) %>% 
    relocate( starts_with("e"), .after=last_col())
  colnames(train) <- c("t","bss",paste("p_",0:k, sep=""),paste("tp_",0:k, sep=""),paste("e_",0:k, sep=""))
  train=train%>%filter( t>(ymd("2015-01-01")+k) )
  
  TN=sum( (train$p_0-train$p_1)^2 )/(nrow(train)-1)
  
  attributes <- colnames(train)
  
  cat(paste0("\tlearn models\n"))
  #on effectue le modèles qui nous permettrons d'effectuer les prédictions
  fff=paste0("p_0 ~ ", paste(attributes[str_detect(attributes, "^p_[1-9]|^tp_|^e_")],collapse="+"))
  lm_p_tp_e <- lm(as.formula(fff), data = train)  
  svr_p_tp_e <- svm(as.formula(fff), data = train,type='eps-regression',kernel='radial')
  rf_p_tp_e <- randomForest(as.formula(fff), data = train, ntree=100)
  
  fff=paste0("p_0 ~ ", paste(attributes[str_detect(attributes, "^p_[1-9]|^tp_")],collapse="+"))
  lm_p_tp <- lm(as.formula(fff), data = train)  
  svr_p_tp <- svm(as.formula(fff), data = train,type='eps-regression',kernel='radial')
  rf_p_tp <- randomForest(as.formula(fff), data = train, ntree=100)
  
  fff=paste0("p_0 ~ ", paste(attributes[str_detect(attributes, "^p_[1-9]")],collapse="+"))
  lm_p <- lm(as.formula(fff), data = train)  
  svr_p <- svm(as.formula(fff), data = train,type='eps-regression',kernel='radial')
  rf_p <- randomForest(as.formula(fff), data = train, ntree=100)
  
  cat(paste0("\tprepare test time series\n"))
  test_bss_pred <-  data_bss %>%filter( (time>=ymd("2020-10-15")-k) &  (time<="2021-01-15") )
  n <- nrow(test_bss_pred)
  
  #On conserve la vérité terrain à part
  gt = test_bss_pred$p
  for( do_approx in c(TRUE,FALSE) ) {
    
    if(do_approx) {
      test_bss_pred[(nrow(test_bss_pred) - h + 1):nrow(test_bss_pred),2] <- approx$tp
      test_bss_pred[(nrow(test_bss_pred) - h + 1):nrow(test_bss_pred),3] <- approx$e
    }
    
    for( classifier in c('lm','svr','rf') ) {
      for( features in c('p','p_tp','p_tp_e') ) {
        model=paste0(classifier,"_",features)
        cat(paste0("\t\ttest model: ",model,", approx: ",do_approx,"\n"))
        #on fait les prévisions avec model
        test_df <- test_bss_pred
        for( i in 1:h ) {
          #On construit le jeu de données
          partial <- list()
          partial[[1]] <- test_df[-(1:k),]
          for (j in 1:(k-1)){
            partial[[j+1]] = (test_df%>%select(p,tp,e))[-c(1:(k-j),(n-j+1):n),]
          } 
          partial[[k+1]] = (test_df%>%select(p,tp,e))[-((n-k+1):n),]
          
          test = bind_cols(partial, .name_repair = ~ vctrs::vec_as_names(..., repair = "unique", quiet = TRUE))
          test = test %>% relocate( starts_with("p"), .after=last_col()) %>% 
            relocate( starts_with("tp"), .after=last_col()) %>% 
            relocate( starts_with("e"), .after=last_col())
          
          colnames(test) <- c("t","bss",paste("p_",0:k, sep=""),paste("tp_",0:k, sep=""),paste("e_",0:k, sep=""))
          test=test%>%filter( t>=ymd("2020-10-15") )
          
          # On effectue la prédiction à i
          test_df$p[k+i] = predict(eval(parse(text=model)), test[i,])
        }
        
        ldf <- data.frame(id_piezo=code_bss[bss_id],
                          id_method_ML=classifier,
                          type="Local", r=k,
                          use_exo_rain=str_detect(features,"tp"),
                          use_exo_eto=str_detect(features,"e"),
                          use_exo_bdlisa=FALSE, approx_exo=do_approx,
                          rmsse=sqrt((mean((as.vector(test_df$p)-as.vector(gt))^2))/TN),
                          mse=mean((as.vector(test_df$p)-as.vector(gt))^2) )
        all_rmsse <- rbind(all_rmsse,ldf)
      }
    }
    
    # #ARIMA
    # 
    # # prediction arima avec p 
    # fit_p <- auto.arima(train$p)
    # fcast <- forecast(fit_p, h=h)
    # ldf <- data.frame(id_piezo=code_bss[bss_id],
    #                   id_method_ML="ARIMA",
    #                   type="Local", r=k,
    #                   use_exo_rain=FALSE,
    #                   use_exo_eto=FALSE,
    #                   use_exo_bdlisa=FALSE, approx_exo=FALSE,
    #                   rmsse=sqrt(mean((as.vector(test$p)-as.vector(fcast[["mean"]]))^2)/TN),
    #                   mse=mean((as.vector(test$p)-as.vector(fcast[["mean"]]))^2)
    # )
    # all_rmsse <- rbind(all_rmsse,ldf)
    # 
    # #prediction arima avec p et tp 
    # fit_p_tp <- auto.arima(train$p, xreg = train$tp)
    # fcast <- forecast(fit_p_tp, xreg = test$tp, h = h)
    # ldf <- data.frame(id_piezo=code_bss[bss_id],
    #                   id_method_ML="ARIMA",
    #                   type="Local", r=NA,
    #                   use_exo_rain=TRUE,
    #                   use_exo_eto=FALSE,
    #                   use_exo_bdlisa=FALSE, approx_exo=FALSE,
    #                   rmsse=sqrt(mean((as.vector(test$p)-as.vector(fcast[["mean"]]))^2)/TN),
    #                   mse=mean((as.vector(test$p)-as.vector(fcast[["mean"]]))^2)
    # )
    # all_rmsse <- rbind(all_rmsse,ldf)
    # #prédiction arima avec p, tp et e 
    # fit_p_tp_e <- auto.arima(train$p, xreg = cbind(train$tp,train$e))
    # fcast <- forecast(fit_p_tp_e, xreg = cbind(test$tp,test$e), h = h)
    # ldf <- data.frame(id_piezo=code_bss[bss_id],
    #                   id_method_ML="ARIMA",
    #                   type="Local", r=NA,
    #                   use_exo_rain=TRUE,
    #                   use_exo_eto=TRUE,
    #                   use_exo_bdlisa=FALSE, approx_exo=FALSE,
    #                   rmsse=sqrt(mean((as.vector(test$p)-as.vector(fcast[["mean"]]))^2)/TN),
    #                   mse=mean((as.vector(test$p)-as.vector(fcast[["mean"]]))^2)
    # )
    # all_rmsse <- rbind(all_rmsse,ldf)
    
    
    #   # prédiction avec prophet
    #   data_bss <-  data %>% filter(bss ==code_bss[bss_id]) %>% select(t,p)
    #   test <- data_bss[(nrow(data_bss) - h+1):(nrow(data_bss)),]
    #   train_proph <-  data_bss[2:(nrow(data_bss) - h),]
    #   names(train_proph) <- c('ds', 'y') 
    #   m <- prophet(train_proph)
    #   future <- make_future_dataframe(m, periods=h)
    #   proph <- predict(m, future)
    #   prev_proph <- proph[(nrow(proph) - h+1):(nrow(proph)),]
    #   
    #   rmsse_prophet[bss_id] <- sqrt((mean((as.vector(test$p)-as.vector(prev_proph$trend))^2))/TN)
    
    write.table(all_rmsse,
                paste0(rep_results,"/",prefix,"local_all_r50.tmp.csv"),
                row.names=FALSE,
                sep="\t",
                dec="." )
  }
}

write.table(all_rmsse,
            paste0(rep_results,"/",prefix,"local_all_r50.csv"),
            row.names=FALSE,
            sep="\t",
            dec="." )

  
  
