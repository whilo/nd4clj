(ns clj-nd4j.core
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
  "Coreces an arbitrary array to an ND$J value"
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
      (if (= (vec shape) (vec (.shape arr)))
        arr ;; array sucessfully created
        nil ;; sometimes ND4J implementations can't create the correct shape.... 
        )))
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
      (if (= shape (.shape arr))
        arr ;; array sucessfully created
        nil ;; sometimes ND4J implementations can't create the correct shape.... 
        )))
  (mp/supports-dimensionality? [m dimensions]
    "Returns true if the implementation supports matrices with the given number of dimensions."
    ;; we support all dimensionalities since we are a full nd-array implementation
    true)

  mp/PDimensionInfo
  (mp/dimensionality [m]
    "Returns the number of dimensions of an array"
    (count (.shape m)))
  (mp/get-shape [m]
    "Returns the shape of the array, typically as a Java array or sequence of dimension sizes.
     Implementations are free to choose what type is used to represent the shape, but it must
     contain only integer values and be traversable as a sequence via clojure.core/seq"
    (.shape m))
  (mp/is-scalar? [m]
    "Tests whether an object is a scalar value, i.e. a value that can exist at a
     specific position in an array."
    ;; An ND4J NDArray is never a scalar (though it may potentially be a 0-dimensional-array?)
    false)
  (mp/is-vector? [m]
    "Tests whether an object is a vector (1D array)"
    (= (count (.shape m)) 1))
  (mp/dimension-count [m dimension-number]
    "Returns the size of a specific dimension. Must throw an exception if the array does not
     have the specified dimension."
    (aget (.shape m) (long dimension-number)))

  mp/PIndexedAccess
  (mp/get-1d [m row]
    (.getDouble m (int row)))
  (mp/get-2d [m row column]
    (.getDouble m (int row) (int column)))
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
    (.mmul m ^org.nd4j.linalg.api.ndarray.INDArray (coerce-nd4j a)))
  (mp/element-multiply [m a]
    (.mul m ^org.nd4j.linalg.api.ndarray.INDArray (coerce-nd4j a)))

  mp/PMatrixProducts
  (mp/inner-product [m a]
    (.mmul m (coerce-nd4j a)))
  (mp/outer-product [m a]
    (coerce-nd4j (mp/outer-product (mp/coerce-param [] m) a))))

(def canonical-object (Nd4j/create 4 2))
(imp/register-implementation canonical-object)

(comment
  (require '[clojure.reflect :refer [reflect]]
           '[clojure.pprint :refer [pprint]])


  (defn into-matrix [vec-of-vecs]
    (->> vec-of-vecs
         (map (partial into-array Double/TYPE))
         into-array
         Nd4j/create))


  (def nd1 (into-matrix [[1 2 3] [4 5 6]]))

  (def nd2 (into-matrix [[1 1] [1 1] [1 1]]))

  (def ndmul (.mmul nd1 nd2))



  (.transpose ndmul)



  (map :name (:members (reflect (new-vector (Nd4j/create 4 2) 3))))

  (def foo (new-vector (Nd4j/create 4 2) 2))

  (seq (.shape (new-vector (Nd4j/create 4 2) 3)))

  (inner-product (construct-matrix foo [1 0])
                 (construct-matrix foo [1 0]))

  (matrix-multiply (construct-matrix foo [[1 0]
                                          [0 1]])
                   (construct-matrix foo [[1 2]
                                          [3 4]]))

  (let [m (construct-matrix foo [[1 0]
                                 [0 1]])]
    (.put m 1 3.0))

  )

