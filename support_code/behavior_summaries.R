libs <- c("dplyr", "readr")
sapply(libs, require, character.only = TRUE)


#User defined variables
behavior <- "freeze" 
row_num <- c(1, 4, 11) #number of rows to consider for each bin. Each row corresponds to 5 minutes. So 1 corresponds to 5 minutes, 4 corresponds to 20 minutes and 11 corresponds to 55 minutes.  
data <- read_csv("/Users/sabnig/Downloads/Lupus_OFMetrics_Freeze_summaries.csv", skip = 2)
data <- data |> select(!"longterm_idx") |> rename(MouseID = exp_prefix)



#No edits needed below this line
cols_to_exclude <- c(paste0(behavior, "_", c("time_no_pred", "time_not_behavior", "time_behavior", "bout_behavior", "not_behavior_dist", "behavior_dist"), sep = ""))
names(data)[!names(data) %in% "MouseID"] <- paste0(behavior, "_", setdiff(names(data), "MouseID"))

data_agg <- lapply(seq_along(row_num), function(x) {
    
    tmp <- data |> group_by(MouseID) |> filter(row_number() <= row_num[x])
    tmpsum <- rowsum(tmp[,sapply(tmp, is.numeric)], tmp$MouseID)
    
    tmpsum <- tmpsum |> mutate(!!paste0("bin_avg_", row_num[x]*5, ".", behavior, "_time_secs") := get(paste0(behavior, "_time_behavior"))/(get(paste0(behavior, "_time_behavior")) + get(paste0(behavior, "_time_not_behavior")))*row_num[x]*5, !!paste0("bin_avg_", row_num[x]*5, ".", behavior, "_distance_cm") := get(paste0(behavior, "_behavior_dist"))/(row_num[x]*5))
    
    tmpsum$MouseID <- rownames(tmpsum)
    tmpsum <- tmpsum |> select(!all_of(cols_to_exclude))
    return(tmpsum)

})


data_res <- Reduce(function(x, y) merge(x, y, by = "MouseID"), data_agg) |> as_tibble()
data_res |> write.csv(paste0(behavior, "_summaries", ".csv"), row.names = FALSE)






