(ns clj-nd4j.core-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix.compliance-tester :as compliance]
            [clj-nd4j.core :refer :all])
  (:import [org.nd4j.linalg.factory Nd4j]))


(deftest compliance-test
  (clojure.core.matrix.compliance-tester/compliance-test (Nd4j/create 4 2)))


(run-tests)
