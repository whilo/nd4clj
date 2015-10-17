(ns nd4clj.core-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix.compliance-tester :as compliance]
            [nd4clj.core :refer :all])
  (:import [org.nd4j.linalg.factory Nd4j]))


(deftest compliance-test
  (clojure.core.matrix.compliance-tester/compliance-test (Nd4j/create 4 2)))


(run-tests)
