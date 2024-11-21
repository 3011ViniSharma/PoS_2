install.packages("gam",repos="http://cran.us.r-project.org")
library(data.table)
library(gam)
loess_prediction = function(raw_loess, lvl1, lvl2, lvl3, time_period, loess_degf, dep, loess_indep_vars){

  #converting empty string to NULL
  if (lvl3 == ''){
  lvl3 <- NULL
  }

  raw_loess <- as.data.table(copy(raw_loess))

  # check NAs in a dataset for modeling
  columns_with_NA = colnames(raw_loess)[sapply(raw_loess, function(x) any(is.na(x)))]
  if (length(columns_with_NA > 0)) {
    print(paste0("NAs are detected in data after preparation step for columns: ",
                 paste(columns_with_NA, collapse = ", ")))
  }
  # replace all infinite values across the full dataset with 0
  raw_loess[sapply(raw_loess, simplify = 'matrix', is.infinite)] <- 0
  # replace NA/NaN across the full dataset with 0
  raw_loess[is.na(raw_loess)] <- 0
  # Selecting span value for gam procedure
  span <- ifelse(loess_degf == 2, 0.5,
               ifelse(loess_degf == 4, 0.29, #0.25
                      ifelse(loess_degf == 6, 0.15, print("Choose loess_degf from set {2, 4, 6}"))
                      )
               )

  # Specifying GAM formula
  gam_formula <- paste("gam(", paste(dep, paste0(
    paste(loess_indep_vars, collapse = " + "), " + ",
    paste("lo(", time_period, ", span = ", span, "), data = ", "raw_loess", sep = "")),
    sep = " ~ " ),
    ")" )

  # Getting the linear trend term
  loess_params = data.table()
  # Calculating parametric and smoothing components of the model
  calc_gam <- function(raw_loess, gam_formula){

    fit <- eval(parse(text = gam_formula))
    if (!is.null(lvl3)){
      unique_lvls = raw_loess[, .N,  by = c(lvl1, lvl2, lvl3)][, N := NULL]
    } else {
      unique_lvls = raw_loess[, .N,  by = c(lvl1, lvl2)][, N := NULL]
    }
    coeffs_dt <- as.data.table(list(names(fit$coefficients), fit$coefficients, unique_lvls))
    setnames(coeffs_dt, c("V1", "V2"), c("parameter", paste0("estimate_", loess_degf)))
    coeffs_dt <- coeffs_dt[substring(parameter, 1, 3) == "lo(", !c("parameter")]
    loess_params <<- rbind(loess_params, coeffs_dt)
    return(as.data.table(list(smoothed = fit$smooth, fitted = fit$fitted.values)))
  }

  # Getting the smoothing trend term
  if (!is.null(lvl3)){
  raw_loess[, c(paste0("loess_p", loess_degf),
              paste0("p_", dep)) := calc_gam(.SD, gam_formula),
              by = .(get(lvl1), get(lvl2), get(lvl3))]
  } else {
    raw_loess[, c(paste0("loess_p", loess_degf),
                  paste0("p_", dep)) := calc_gam(.SD, gam_formula),
                  by = .(get(lvl1), get(lvl2))]
  }
  # Calculating the full LOESS line and adding results to the main dataset
  base_loess <- merge(raw_loess, loess_params,  by = c(lvl2, lvl1, lvl3), all.x = TRUE)
  base_loess[, "trend_term" := get(paste0("loess_p", loess_degf)) +
             get(paste0("estimate_", loess_degf)) * get(time_period)]
  base_loess[, paste0("loess_predicted_", dep) := exp(get(paste0("p_", dep)))]

  # Checking, that trend_term has finite value
  base_loess <- base_loess[is.nan(get("trend_term")) |
                           is.infinite(get("trend_term")),
                           "trend_term" := week_number/100]

  return(base_loess)
}
