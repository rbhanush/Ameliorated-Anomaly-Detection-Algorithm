function [F1,precision,recall,accuracy] = printClassMetrics (pred_val , yval)
  verbose = 1;
  accuracy = mean(double(pred_val == yval));
  acc_all0 = mean(double(0 == yval));
  if (verbose)
    fprintf("|--> accuracy == %f vs accuracy_all0 == %f \n",accuracy,acc_all0);
  end

  actual_positives = sum(yval == 1);
  actual_negatives = sum(yval == 0);
  true_positives = sum((pred_val == 1) & (yval == 1));
  false_positives = sum((pred_val == 1) & (yval == -1));
  false_negatives = sum((pred_val == -1) & (yval == 1));
  precision = 0; 
  if ( (true_positives + false_positives) > 0)
    precision = true_positives / (true_positives + false_positives);
  end
  
  recall = 0; 
  if ( (true_positives + false_negatives) > 0 )
    recall = true_positives / (true_positives + false_negatives);
  end

  F1 = 0; 
  if ( (precision + recall) > 0) 
    F1 = 2 * precision * recall / (precision + recall);
  end
  
  outlier = sum((yval == -1));
  outlier_a = sum((pred_val==-1)&(yval==-1));
  outlier_accu = double(outlier_a/outlier);
 
  if (verbose) 
    fprintf("|-->  true_positives == %i  (actual positive =%i) \n",true_positives,actual_positives);
    fprintf("|-->  false_positives == %i \n",false_positives);
    fprintf("|-->  false_negatives == %i \n",false_negatives);
    fprintf("|-->  precision == %f \n",precision);
    fprintf("|-->  recall == %f \n",recall);
    fprintf("|-->  F1 == %f \n",F1);
    fprintf("|-->  outlier accuracy is == %f \n",outlier_accu);
  end 
  
end
