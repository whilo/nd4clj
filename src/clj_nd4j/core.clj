(ns clj-nd4j.core
  (:require [clojure.core.matrix.protocols :refer :all]
            [clojure.core.matrix.implementations :as imp])
  (:import [org.nd4j.linalg.factory Nd4j]))


(defn vs? [m] (or (vector? m) (seq? m)))


(extend-type org.nd4j.linalg.api.ndarray.INDArray
  PImplementation
  (implementation-key [m] :nd4j)
  (meta-info [m] {:doc "nd4j implementation of the core.matrix
  protocols. Allows different backends, e.g. jblas or jcublas for
  graphic cards."})
  (construct-matrix [m data]
    "Returns a new n-dimensional array containing the given data. data should be in the form of either
     nested sequences or a valid existing array.

     The return value should be in the preferred format of the given implementation. If the implementation
     does not support the required dimensionality or element type then it may either:
      - Throw an error
      - Return nil to indicate that a default implementation should be used instead

     0-dimensional arrays / scalars are permitted."
    (if (vs? data)
      (if (vs? (first data))
        (if (vs? (first (first data)))
          (throw (ex-info "Only 2-dimensional tensors are allowed."
                          {:data data}))
          (->> data
               (map (partial into-array Double/TYPE))
               into-array
               Nd4j/create))
        (Nd4j/create (double-array data)))
      data))
  (new-vector [m length]
    "Returns a new vector (1D column matrix) of the given length, filled with numeric zero."
    (Nd4j/create length))
  (new-matrix [m rows columns]
    "Returns a new matrix (regular 2D matrix) with the given number of rows and columns, filled with numeric zero."
    (Nd4j/create rows columns))
  (new-matrix-nd [m shape]
    "Returns a new general matrix of the given shape.
     Must return nil if the shape is not supported by the implementation.
     Shape must be a sequence of dimension sizes."
    (let [[a b c] shape]
      (if c nil
          (if b (Nd4j/create a b)
              (if a (Nd4j/create a)
                  (throw (ex-info "Don't know how to create empty shape."
                                  {:shape shape})))))))
  (supports-dimensionality? [m dimensions]
    "Returns true if the implementation supports matrices with the given number of dimensions."
    (or (= dimensions 1) (= dimensions 2)))

  PDimensionInfo
  (dimensionality [m]
    "Returns the number of dimensions of an array"
    (count (.shape m)))
  (get-shape [m]
    "Returns the shape of the array, typically as a Java array or sequence of dimension sizes.
     Implementations are free to choose what type is used to represent the shape, but it must
     contain only integer values and be traversable as a sequence via clojure.core/seq"
    (.shape m))
  (is-scalar? [m]
    "Tests whether an object is a scalar value, i.e. a value that can exist at a
     specific position in an array."
    (= (count (.shape m)) 0))
  (is-vector? [m]
    "Tests whether an object is a vector (1D array)"
    (= (count (.shape m)) 1))
  (dimension-count [m dimension-number]
    "Returns the size of a specific dimension. Must throw an exception if the array does not
     have the specified dimension."
    (nth (.shape m) dimension-number))

  PIndexedAccess
  (get-1d [m row]
    (.getDouble m row))
  (get-2d [m row column]
    (.getDouble m row column))
  (get-nd [m indexes]
    (let [ci (count indexes)
          [r c] indexes]
      (case ci
        1 (.getDouble m r)
        2 (.getDouble m r c)
        (throw (ex-info "nd4j only supports 2 dimensional matrices."
                        {:indexes indexes
                         :matrix m})))))


  PIndexedSetting
  (set-1d [m row v]
    (let [d (.dup m)]
      (.put d row v)))
  (set-2d [m row column v]
    (let [d (.dup m)]
      (.put d row column v)))
  (set-nd [m indexes v]
    (let [d (.dup m)
          ci (count indexes)
          [r c] indexes]
      (case ci
        1 (.put d r v)
        2 (.put d r c v)
        (throw (ex-info "nd4j only supports 2 dimensional matrices."
                        {:indexes indexes
                         :matrix m})))))
  (is-mutable? [m] false)

  PTypeInfo
  (element-type [m] Double/TYPE)

  PMatrixMultiply
  (matrix-multiply [m a]
    (.mmul m (if (isa? (type a) org.nd4j.linalg.api.ndarray.INDArray)
               a (construct-matrix m a))))
  (element-multiply [m a]
    (.mul m a))

  PMatrixProducts
  (inner-product [m a]
    (.mmul m (.transpose (if (isa? (type a) org.nd4j.linalg.api.ndarray.INDArray)
                           a (construct-matrix m a)))))
  (outer-product [m a]
    (.mmul (.transpose m) (if (isa? (type a) org.nd4j.linalg.api.ndarray.INDArray)
                            a (construct-matrix m a)))))


(imp/register-implementation (Nd4j/create 4 2))

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
