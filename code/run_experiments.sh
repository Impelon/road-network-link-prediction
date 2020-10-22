for i in {0..4}
do
  echo "Running 'tu_berlin_simplified' on iteration $i"
  python3 predict_links.py "experiments/tu_berlin_simplified/$i-30/" "TU Berlin, Berlin, Deutschland"
done

for i in {0..4}
do
  echo "Running 'u_frankfurter_tor_simplified' on iteration $i"
  python3 predict_links.py "experiments/u_frankfurter_tor_simplified/$i-30/" "U Frankfurter Tor, Berlin, Deutschland"
done
# for i in {0..4}
# do
#   echo "Running on iteration $i"
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i/random_forest_classifier_small.joblib" "experiments/u_frankfurter_tor_simplified/$i/" "random_forest_classifier_cross_small"
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i/random_forest_classifier_large.joblib" "experiments/u_frankfurter_tor_simplified/$i/" "random_forest_classifier_cross_large"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i/random_forest_classifier_small.joblib" "experiments/tu_berlin_simplified/$i/" "random_forest_classifier_cross_small"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i/random_forest_classifier_large.joblib" "experiments/tu_berlin_simplified/$i/" "random_forest_classifier_cross_large"
#
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i-30/random_forest_classifier_small.joblib" "experiments/u_frankfurter_tor_simplified/$i-30/" "random_forest_classifier_cross_30_small"
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i-30/random_forest_classifier_large.joblib" "experiments/u_frankfurter_tor_simplified/$i-30/" "random_forest_classifier_cross_30_large"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i-30/random_forest_classifier_small.joblib" "experiments/tu_berlin_simplified/$i-30/" "random_forest_classifier_cross_30_small"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i-30/random_forest_classifier_large.joblib" "experiments/tu_berlin_simplified/$i-30/" "random_forest_classifier_cross_30_large"
#
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i-30/random_forest_classifier_small.joblib" "experiments/u_frankfurter_tor_simplified/$i/" "random_forest_classifier_cross_30_small"
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i-30/random_forest_classifier_large.joblib" "experiments/u_frankfurter_tor_simplified/$i/" "random_forest_classifier_cross_30_large"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i-30/random_forest_classifier_small.joblib" "experiments/tu_berlin_simplified/$i/" "random_forest_classifier_cross_30_small"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i-30/random_forest_classifier_large.joblib" "experiments/tu_berlin_simplified/$i/" "random_forest_classifier_cross_30_large"
#
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i/random_forest_classifier_small.joblib" "experiments/u_frankfurter_tor_simplified/$i-30/" "random_forest_classifier_cross_small"
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i/random_forest_classifier_large.joblib" "experiments/u_frankfurter_tor_simplified/$i-30/" "random_forest_classifier_cross_large"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i/random_forest_classifier_small.joblib" "experiments/tu_berlin_simplified/$i-30/" "random_forest_classifier_cross_small"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i/random_forest_classifier_large.joblib" "experiments/tu_berlin_simplified/$i-30/" "random_forest_classifier_cross_large"
#
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i/random_forest_classifier_small.joblib" "experiments/tu_berlin_simplified/$i-30/" "random_forest_classifier_small"
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i/random_forest_classifier_large.joblib" "experiments/tu_berlin_simplified/$i-30/" "random_forest_classifier_large"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i/random_forest_classifier_small.joblib" "experiments/u_frankfurter_tor_simplified/$i-30/" "random_forest_classifier_small"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i/random_forest_classifier_large.joblib" "experiments/u_frankfurter_tor_simplified/$i-30/" "random_forest_classifier_large"
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i-30/random_forest_classifier_small.joblib" "experiments/tu_berlin_simplified/$i/" "random_forest_classifier_30_small"
#   python3 classifier_predict_links.py "experiments/tu_berlin_simplified/$i-30/random_forest_classifier_large.joblib" "experiments/tu_berlin_simplified/$i/" "random_forest_classifier_30_large"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i-30/random_forest_classifier_small.joblib" "experiments/u_frankfurter_tor_simplified/$i/" "random_forest_classifier_30_small"
#   python3 classifier_predict_links.py "experiments/u_frankfurter_tor_simplified/$i-30/random_forest_classifier_large.joblib" "experiments/u_frankfurter_tor_simplified/$i/" "random_forest_classifier_30_large"
# done
