#pragma once
#include<tuple>
#include<iterator>
#include<utility>
// Code taken from https://codereview.stackexchange.com/questions/30846/zip-like-functionality-with-c11s-range-based-for-loop
//
/***************************
// helper for tuple_subset and tuple_tail (from http://stackoverflow.com/questions/8569567/get-part-of-stdtuple)
***************************/
template <size_t... n>
struct ct_integers_list {
    template <size_t m>
    struct push_back
    {
        typedef ct_integers_list<n..., m> type;
    };
};

template <size_t max>
struct ct_iota_1
{
    typedef typename ct_iota_1<max-1>::type::template push_back<max>::type type;
};

template <>
struct ct_iota_1<0>
{
    typedef ct_integers_list<> type;
};

/***************************
// return a subset of a tuple
***************************/
#if 0
template <size_t... indices, typename Tuple>
auto tuple_subset(const Tuple& tpl, ct_integers_list<indices...>)
    -> decltype(std::make_tuple(std::get<indices>(tpl)...))
{
    return std::make_tuple(std::get<indices>(tpl)...);
    // this means:
    //   make_tuple(get<indices[0]>(tpl), get<indices[1]>(tpl), ...)
}
#else
template <size_t... indices, typename Tuple>
auto tuple_subset(Tuple&& tpl, ct_integers_list<indices...>)
        -> decltype(std::make_tuple(std::get<indices>(std::forward<Tuple>(tpl))...))
{
    return std::make_tuple(std::get<indices>(std::forward<Tuple>(tpl))...);
    // this means:
    //     make_tuple(get<indices[0]>(tpl), get<indices[1]>(tpl), ...)
}
#endif

/***************************
// return the tail of a tuple
***************************/
template <typename Head, typename... Tail>
inline std::tuple<Tail...> tuple_tail(const std::tuple<Head, Tail...>& tpl)
{
    return tuple_subset(tpl, typename ct_iota_1<sizeof...(Tail)>::type());
    // this means:
    //   tuple_subset<1, 2, 3, ..., sizeof...(Tail)-1>(tpl, ..)
}

/***************************
// increment every element in a tuple (that is referenced)
***************************/
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
increment(std::tuple<Tp...>& t)
{ }

#if 0
template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<(I < sizeof...(Tp)), void>::type
increment(std::tuple<Tp...>& t)
{
    std::get<I>(t)++ ;
    increment<I + 1, Tp...>(t);
}
#else
template<std::size_t I = 0, typename... Tp,
    typename = typename std::enable_if<I < sizeof...(Tp), void>::type>
inline void increment(std::tuple<Tp...>& t)
{
    std::get<I>(t)++ ;
    increment<I + 1, Tp...>(t);
}
#endif

/**************************** 
// check equality of a tuple
****************************/
template<typename T1>
inline bool not_equal_tuples( const std::tuple<T1>& t1,  const std::tuple<T1>& t2 )
{
    return (std::get<0>(t1) != std::get<0>(t2));
}

template<typename T1, typename... Ts>
inline bool not_equal_tuples( const std::tuple<T1, Ts...>& t1,  const std::tuple<T1, Ts...>& t2 )
{
    return (std::get<0>(t1) != std::get<0>(t2)) && not_equal_tuples( tuple_tail(t1), tuple_tail(t2) );
}

/**************************** 
// dereference a subset of elements of a tuple (dereferencing the iterators)
****************************/
template <size_t... indices, typename Tuple>
auto dereference_subset(const Tuple& tpl, ct_integers_list<indices...>)
    -> decltype(std::tie(*std::get<indices-1>(tpl)...))
{
    return std::tie(*std::get<indices-1>(tpl)...);
}

/**************************** 
// dereference every element of a tuple (applying operator* to each element, and returning the tuple)
****************************/
template<typename... Ts>
inline auto
  dereference_tuple(std::tuple<Ts...>& t1) -> decltype( dereference_subset( std::tuple<Ts...>(), typename ct_iota_1<sizeof...(Ts)>::type()))
  {
    return dereference_subset( t1, typename ct_iota_1<sizeof...(Ts)>::type());
  }


template< typename T1, typename... Ts >
class zipper
{
    public:

    class iterator : std::iterator<std::forward_iterator_tag, std::tuple<typename T1::value_type, typename Ts::value_type...> >
    {
        protected:
            std::tuple<typename T1::iterator, typename Ts::iterator...> current;
        public:

        explicit iterator(  typename T1::iterator s1, typename Ts::iterator... s2 ) : 
            current(s1, s2...) {};

        iterator( const iterator& rhs ) :  current(rhs.current) {};

        iterator& operator++() {
            increment(current);
            return *this;
        }

        iterator operator++(int) {
            auto a = *this;
            increment(current);
            return a;
        }

        bool operator!=( const iterator& rhs ) {
            return not_equal_tuples(current, rhs.current);
        }

        typename iterator::value_type operator*() {
            return dereference_tuple(current);
        }
    };


    explicit zipper( T1& a, Ts&... b):
                        begin_( a.begin(), (b.begin())...), 
                        end_( a.end(), (b.end())...) {};

    zipper(const zipper<T1, Ts...>& a) :
                        begin_(  a.begin_ ), 
                        end_( a.end_ ) {};

    template<typename U1, typename... Us>
    zipper<U1, Us...>& operator=( zipper<U1, Us...>& rhs) {
        begin_ = rhs.begin_;
        end_ = rhs.end_;
        return *this;
    }

    zipper<T1, Ts...>::iterator& begin() {
        return begin_;
    }

    zipper<T1, Ts...>::iterator& end() {
        return end_;
    }

    zipper<T1, Ts...>::iterator begin_;
    zipper<T1, Ts...>::iterator end_;
};



//from cppreference.com: 
template <class T>
  struct special_decay
  {
     using type = typename std::decay<T>::type;
  };

//allows the use of references:
template <class T>
 struct special_decay<std::reference_wrapper<T>>
 {
   using type = T&;
 };

template <class T>
 using special_decay_t = typename special_decay<T>::type;

//allows template type deduction for zipper:
template <class... Types>
 zipper<special_decay_t<Types>...> zip(Types&&... args)
 {
   return zipper<special_decay_t<Types>...>(std::forward<Types>(args)...);
 }

