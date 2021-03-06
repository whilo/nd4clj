(ns nd4clj.core
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.implementations :as imp])
  (:import [org.nd4j.linalg.factory Nd4j]))

;; TODO: eliminate reflection warnings
(set! *warn-on-reflection* true)
;; (set! *unchecked-math* true)

(declare canonical-object)

(defn vs? [m] (or (vector? m) (seq? m)))

(defn coerce-nd4j
  "Coreces an arbitrary array to an ND4J value"
  (^org.nd4j.linalg.api.ndarray.INDArray [a]
    (if (instance? org.nd4j.linalg.api.ndarray.INDArray a)
               a (mp/construct-matrix canonical-object a))))

(defn- column-major-strides
  ^ints [shape]
  (let [shape (int-array shape)
        n (alength shape)
        strides (int-array n)]
    (areduce shape i st 1
             (do
               (aset strides i (int st))
               (* st (aget shape i))))
    strides))

(defn- row-major-strides
  ^ints [shape]
  (let [shape (int-array shape)
        n (alength shape)
        strides (int-array n)]
    (areduce shape i st 1
             (let [ix (- (dec n) i)]
               (aset strides ix (int st))
               (* st (aget shape ix))))
    strides))

(defn get-shape [^org.nd4j.linalg.api.ndarray.INDArray m]
  (let [s (.shape m)]
    (vec s)
    #_(cond (.isScalar m) []
          (.isVector m) (vec (rest s))
          :else (vec s))))

#_(let [shape (int-array [1 4])
      dbs (m/to-double-array [[1 2 3 4]])]
  (.isVector (Nd4j/create dbs (int-array shape) (row-major-strides shape) 0 \c)))

(extend-type org.nd4j.linalg.api.ndarray.INDArray
  mp/PImplementation
  (mp/implementation-key [m] :nd4j)
  (mp/meta-info [m] {:doc "nd4j implementation of the core.matrix
  protocols. Allows different backends, e.g. jblas or jcublas for
  graphic cards."})
  (mp/construct-matrix [m data]
    "Returns a new n-dimensional array containing the given data. data should be in the form of either
     nested sequences or a valid existing array.

     The return value should be in the preferred format of the given implementation. If the implementation
     does not support the required dimensionality or element type then it may either:
      - Throw an error
      - Return nil to indicate that a default implementation should be used instead

     0-dimensional arrays / scalars are permitted."
    (let [^ints shape (int-array (m/shape data))
          ;; _ (println (str "Creating ND4J array of shape: " (vec shape)))
          ^doubles dbs (m/to-double-array data)
          arr (Nd4j/create dbs (int-array shape) (row-major-strides shape) 0 \c)]
      #_(.setShape arr shape)
      (when-not (= (vec shape) (get-shape arr))
        (throw (ex-info "Shape mismatch" {:data data
                                          :shape (vec shape)
                                          :actual-shape [(get-shape arr) (vec (.shape arr))]})))

      arr ;; array sucessfully created
      ;;        nil ;; sometimes ND4J implementations can't create the correct shape....
      ))
  (mp/new-vector [m length]
    "Returns a new vector (1D column matrix) of the given length, filled with numeric zero."
    (Nd4j/create (int length)))
  (mp/new-matrix [m rows columns]
    "Returns a new matrix (regular 2D matrix) with the given number of rows and columns, filled with numeric zero."
    (Nd4j/create (int rows) (int columns)))
  (mp/new-matrix-nd [m shape]
    "Returns a new general matrix of the given shape.
     Must return nil if the shape is not supported by the implementation.
     Shape must be a sequence of dimension sizes."
    (let [^ints shape (int-array shape)
          ;; _ (println (str "Creating ND4J array of shape: " (vec shape)))
          arr (Nd4j/create shape)]
      #_(.setShape arr shape)
      ;; sometimes ND4J implementations can't create the correct shape....
      (when (= shape (mp/get-shape arr))
        (throw (ex-info "Wrong shape created:" {:shape shape
                                                :actual (mp/get-shape arr)})))
      arr ;; array sucessfully created
      ))
  (mp/supports-dimensionality? [m dimensions]
    "Returns true if the implementation supports matrices with the given number of dimensions."
    ;; we support all dimensionalities since we are a full nd-array implementation
    true)

  mp/PDimensionInfo
  (mp/dimensionality [m]
    "Returns the number of dimensions of an array"
    (count (get-shape m)))
  (mp/get-shape [m]
    "Returns the shape of the array, typically as a Java array or sequence of dimension sizes.
     Implementations are free to choose what type is used to represent the shape, but it must
     contain only integer values and be traversable as a sequence via clojure.core/seq"
    (get-shape m))
  (mp/is-scalar? [m]
    "Tests whether an object is a scalar value, i.e. a value that can exist at a
     specific position in an array."
    ;; An ND4J NDArray is never a scalar (though it may potentially be a 0-dimensional-array?)
    (zero? (count (get-shape m))))
  (mp/is-vector? [m]
    "Tests whether an object is a vector (1D array)"
    (= (count (get-shape m)) 1))
  (mp/dimension-count [m dimension-number]
    "Returns the size of a specific dimension. Must throw an exception if the array does not
     have the specified dimension."
    (let [shape (get-shape m)
          c (count shape)]
      (when (> dimension-number c)
        (throw (ex-info "Dimension out of range." {:shape shape})))
      (get shape (long dimension-number))))

  ;; TODO: for some reason the [row] and [row,column] accessors trigger an ND4J bug?
  mp/PIndexedAccess
  (mp/get-1d [m row]
    (let [ixs (int-array [row])]
      (.getDouble m ixs)))
  (mp/get-2d [m row column]
    (let [ixs (int-array [row column])]
      (.getDouble m ixs)))
  (mp/get-nd [m indexes]
    (let [ixs (int-array indexes)]
      (.getDouble m ixs)))

  mp/PIndexedSetting
  (mp/set-1d [m row v]
    (let [d (.dup m)
          ixs (int-array [row])]
      (.putScalar d ixs (double v))))
  (mp/set-2d [m row column v]
    (let [d (.dup m)
          ixs (int-array [row column])]
      (.putScalar d ixs (double v))))
  (mp/set-nd [m indexes v]
    (let [d (.dup m)
          indexes (int-array indexes)]
      (.putScalar d indexes (double v))))
  (mp/is-mutable? [m] false)

  mp/PTypeInfo
  (mp/element-type [m] Double/TYPE)

  mp/PMatrixMultiply
  (mp/matrix-multiply [m a]
    (let [^org.nd4j.linalg.api.ndarray.INDArray a (coerce-nd4j a)]
      (cond (and (.isRowVector m) (.isRowVector a))
            (.mmul m (.transpose a))

            :else
            (.mmul m a)
            #_(and (.isRowVector m) (.isColumnVector a))
            )))
  (mp/element-multiply [m a]
    (let [^org.nd4j.linalg.api.ndarray.INDArray a (coerce-nd4j a)]
      (.mul m a)))

  mp/PMatrixProducts
  (mp/inner-product [m a]
    (.mmul m (coerce-nd4j a)))
  (mp/outer-product [m a]
    (coerce-nd4j (mp/outer-product (mp/coerce-param [] m) a))))

(def canonical-object (Nd4j/create 4 2))
(imp/register-implementation canonical-object)

(comment
  (require '[clojure.reflect :refer [reflect]]
           '[clojure.pprint :refer [pprint]]
           '[clojure.core.matrix :refer [matrix mmul] :as mat]
           '[clojure.core.matrix.compliance-tester :as ct])


  (defn into-matrix [vec-of-vecs]
    (->> vec-of-vecs
         (map (partial into-array Double/TYPE))
         into-array
         Nd4j/create))


  (def nd1 (into-matrix [[1 2 3] [4 5 6]]))

  (def nd2 (into-matrix [[1 1] [1 1] [1 1]]))

  (get-shape nd2)


  (def ndmul (.mmul nd1 nd2))

  (mat/mmul (mat/matrix [[2 0] [0 1] [3 0]])
            (mat/matrix [[1 0 0] [0 1 0]]))

  (.transpose ndmul)

  (get-shape (mp/construct-matrix canonical-object [1 0] ))

  (get-shape (.reshape (mp/construct-matrix canonical-object [[1 0]] ) (int-array [2])))

  (clojure.core.matrix/set-current-implementation :nd4j)


  (mmul (matrix [1 2 3]) (matrix [1 2 3])) ; vector times vector
                                        ; => 14
  (mmul (matrix [[1 2 3]]) (matrix [[1] [2] [3]])) ; multiplication of matrices as row and column vectors
                                        ; => [[14]]
  (mmul (matrix [1 2 3]) (matrix [[1] [2] [3]])) ; vector times column vector
                                        ; => [14]
  (mmul (matrix [[1 2 3]]) (matrix [1 2 3])) ; row vector times vector
                                        ; => [14]
  (mmul (matrix [[1 2 3][4 5 6][7 8 9]]) (matrix [[1] [2] [3]])) ; matrix times column vector
                                        ; => [[14] [32] [50]]
  (mmul (matrix [[1 2 3][4 5 6][7 8 9]]) (matrix [1 2 3])) ; matrix times vector
                                        ; => [14 32 50]
  (mmul (matrix [1 2 3]) (matrix [[1 2 3]])) ; vector treated as row vector times row vector
                                        ; => RuntimeException Mismatched vector sizes ...
  (mmul (matrix [[1] [2] [3]]) (matrix [1 2 3])) ; column vector times
                                        ; vector treated as column vector => RuntimeException Mismatched
                                        ; vector sizes ...


  (ct/compliance-test (mat/matrix [[1 0] [0 1]]))

  (let [{:keys [shape actual-shape]} (ex-data *e)]
    (= shape actual-shape))

  (map :name (:members (reflect (new-vector (Nd4j/create 4 2) 3))))

  (def foo (new-vector (Nd4j/create 4 2) 2))

  (seq (.shape (new-vector (Nd4j/create 4 2) 3)))

  (inner-product (construct-matrix foo [1 0])
                 (construct-matrix foo [1 0]))

  (matrix-multiply (construct-matrix foo [[1 0]
                                          [0 1]])
                   (construct-matrix foo [[1 2]
                                          [3 4]]))

  (seq (.shape (Nd4j/create (double-array [1 2]) (int-array [2 1 1]) (int-array [1]) 0 \c)))

  (vec (.shape (Nd4j/create 1 4)))

  (let [m (construct-matrix foo [[1 0]
                                 [0 1]])]
    (.put m 1 3.0)))
