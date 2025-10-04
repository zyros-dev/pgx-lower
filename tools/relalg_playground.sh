#!/bin/bash

MLIR='module {
  func.func @main() -> !dsa.table {
    %0 = relalg.basetable  {column_order = ["n_nationkey", "n_name", "n_regionkey", "n_comment"], table_identifier = "nation|oid:26581790"} columns: {n_comment => @nation::@n_comment({type = !db.string}), n_name => @nation::@n_name({type = !db.string}), n_nationkey => @nation::@n_nationkey({type = i32}), n_regionkey => @nation::@n_regionkey({type = i32})}
    %1 = relalg.basetable  {column_order = ["r_regionkey", "r_name", "r_comment"], table_identifier = "region|oid:26581785"} columns: {r_comment => @region::@r_comment({type = !db.string}), r_name => @region::@r_name({type = !db.string}), r_regionkey => @region::@r_regionkey({type = i32})}
    %2 = relalg.selection %1 (%arg0: !relalg.tuple){
      %20 = relalg.getcol %arg0 @region::@r_name : !db.string
      %21 = db.constant("EUROPE") : !db.string
      %22 = db.constant("EUROPE                   ") : !db.string
      %23 = db.compare eq %20 : !db.string, %22 : !db.string
      relalg.return %23 : i1
    }
    %3 = relalg.join %0, %2 (%arg0: !relalg.tuple){
      %20 = relalg.getcol %arg0 @nation::@n_regionkey : i32
      %21 = relalg.getcol %arg0 @region::@r_regionkey : i32
      %22 = db.compare eq %20 : i32, %21 : i32
      relalg.return %22 : i1
    }
    %4 = relalg.projection all [@nation::@n_name,@nation::@n_nationkey] %3
    %5 = relalg.basetable  {column_order = ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"], table_identifier = "supplier|oid:26581800"} columns: {s_acctbal => @supplier::@s_acctbal({type = !db.decimal<12, 2>}), s_address => @supplier::@s_address({type = !db.string}), s_comment => @supplier::@s_comment({type = !db.string}), s_name => @supplier::@s_name({type = !db.string}), s_nationkey => @supplier::@s_nationkey({type = i32}), s_phone => @supplier::@s_phone({type = !db.string}), s_suppkey => @supplier::@s_suppkey({type = i32})}
    %6 = relalg.join %4, %5 (%arg0: !relalg.tuple){
      %20 = relalg.getcol %arg0 @nation::@n_nationkey : i32
      %21 = relalg.getcol %arg0 @supplier::@s_nationkey : i32
      %22 = db.compare eq %20 : i32, %21 : i32
      relalg.return %22 : i1
    }
    %7 = relalg.projection all [@supplier::@s_acctbal,@supplier::@s_name,@supplier::@s_address,@supplier::@s_phone,@supplier::@s_comment,@supplier::@s_suppkey,@nation::@n_name] %6
    %8 = relalg.basetable  {column_order = ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"], table_identifier = "part|oid:26581795"} columns: {p_brand => @part::@p_brand({type = !db.string}), p_comment => @part::@p_comment({type = !db.string}), p_container => @part::@p_container({type = !db.string}), p_mfgr => @part::@p_mfgr({type = !db.string}), p_name => @part::@p_name({type = !db.string}), p_partkey => @part::@p_partkey({type = i32}), p_retailprice => @part::@p_retailprice({type = !db.decimal<12, 2>}), p_size => @part::@p_size({type = i32}), p_type => @part::@p_type({type = !db.string})}
    %9 = relalg.selection %8 (%arg0: !relalg.tuple){
      %20 = relalg.getcol %arg0 @part::@p_type : !db.string
      %21 = db.constant("%BRASS") : !db.string
      %22 = db.runtime_call "Like"(%20, %21) : (!db.string, !db.string) -> i1
      %23 = relalg.getcol %arg0 @part::@p_size : i32
      %c15_i32 = arith.constant 15 : i32
      %24 = db.compare eq %23 : i32, %c15_i32 : i32
      %25 = db.and %22, %24 : i1, i1
      relalg.return %25 : i1
    }
    %10 = relalg.basetable  {column_order = ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"], table_identifier = "partsupp|oid:26581805"} columns: {ps_availqty => @partsupp::@ps_availqty({type = i32}), ps_comment => @partsupp::@ps_comment({type = !db.string}), ps_partkey => @partsupp::@ps_partkey({type = i32}), ps_suppkey => @partsupp::@ps_suppkey({type = i32}), ps_supplycost => @partsupp::@ps_supplycost({type = !db.decimal<12, 2>})}
    %11 = relalg.selection %10 (%arg0: !relalg.tuple){
      %20 = relalg.getcol %arg0 @partsupp::@ps_partkey : i32
      %21 = relalg.getcol %arg0 @part::@p_partkey : i32
      %22 = db.compare eq %20 : i32, %21 : i32
      %23 = relalg.getcol %arg0 @partsupp::@ps_suppkey : i32
      %24 = relalg.getcol %arg0 @supplier::@s_suppkey : i32
      %25 = db.compare eq %23 : i32, %24 : i32
      %26 = db.and %22, %25 : i1, i1
      relalg.return %26 : i1
    }
    %12 = relalg.selection %11 (%arg0: !relalg.tuple){
      %20 = relalg.getcol %arg0 @partsupp::@ps_supplycost : !db.decimal<12, 2>
      %21 = relalg.basetable  {column_order = ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"], table_identifier = "supplier|oid:26581800"} columns: {s_acctbal => @supplier1::@s_acctbal({type = !db.decimal<12, 2>}), s_address => @supplier1::@s_address({type = !db.string}), s_comment => @supplier1::@s_comment({type = !db.string}), s_name => @supplier1::@s_name({type = !db.string}), s_nationkey => @supplier1::@s_nationkey({type = i32}), s_phone => @supplier1::@s_phone({type = !db.string}), s_suppkey => @supplier1::@s_suppkey({type = i32})}
      %22 = relalg.basetable  {column_order = ["n_nationkey", "n_name", "n_regionkey", "n_comment"], table_identifier = "nation|oid:26581790"} columns: {n_comment => @nation1::@n_comment({type = !db.string}), n_name => @nation1::@n_name({type = !db.string}), n_nationkey => @nation1::@n_nationkey({type = i32}), n_regionkey => @nation1::@n_regionkey({type = i32})}
      %23 = relalg.basetable  {column_order = ["r_regionkey", "r_name", "r_comment"], table_identifier = "region|oid:26581785"} columns: {r_comment => @region1::@r_comment({type = !db.string}), r_name => @region1::@r_name({type = !db.string}), r_regionkey => @region1::@r_regionkey({type = i32})}
      %24 = relalg.selection %23 (%arg1: !relalg.tuple){
        %39 = relalg.getcol %arg1 @region1::@r_name : !db.string
        %40 = db.constant("EUROPE") : !db.string
        %41 = db.constant("EUROPE                   ") : !db.string
        %42 = db.compare eq %39 : !db.string, %41 : !db.string
        relalg.return %42 : i1
      }
      %25 = relalg.join %22, %24 (%arg1: !relalg.tuple){
        %39 = relalg.getcol %arg1 @nation1::@n_regionkey : i32
        %40 = relalg.getcol %arg1 @region1::@r_regionkey : i32
        %41 = db.compare eq %39 : i32, %40 : i32
        relalg.return %41 : i1
      }
      %26 = relalg.projection all [@nation1::@n_nationkey] %25
      %27 = relalg.join %21, %26 (%arg1: !relalg.tuple){
        %39 = relalg.getcol %arg1 @supplier1::@s_nationkey : i32
        %40 = relalg.getcol %arg1 @nation1::@n_nationkey : i32
        %41 = db.compare eq %39 : i32, %40 : i32
        relalg.return %41 : i1
      }
      %28 = relalg.projection all [@supplier1::@s_suppkey] %27
      %29 = relalg.basetable  {column_order = ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"], table_identifier = "partsupp|oid:26581805"} columns: {ps_availqty => @partsupp1::@ps_availqty({type = i32}), ps_comment => @partsupp1::@ps_comment({type = !db.string}), ps_partkey => @partsupp1::@ps_partkey({type = i32}), ps_suppkey => @partsupp1::@ps_suppkey({type = i32}), ps_supplycost => @partsupp1::@ps_supplycost({type = !db.decimal<12, 2>})}
      %30 = relalg.selection %29 (%arg1: !relalg.tuple){
        %39 = relalg.getcol %arg1 @partsupp1::@ps_partkey : i32
        %40 = relalg.getcol %arg0 @part::@p_partkey : i32
        %41 = db.compare eq %39 : i32, %40 : i32
        %42 = relalg.getcol %arg1 @partsupp1::@ps_suppkey : i32
        %43 = relalg.getcol %arg0 @supplier::@s_suppkey : i32
        %44 = db.compare eq %42 : i32, %43 : i32
        %45 = db.and %41, %44 : i1, i1
        relalg.return %45 : i1
      }
      %31 = relalg.join %28, %30 (%arg1: !relalg.tuple){
        %true = arith.constant true
        relalg.return %true : i1
      }
      %32 = relalg.projection all [@partsupp1::@ps_supplycost] %31
      %33 = relalg.aggregation %32 [] computes : [@aggr0::@min({type = !db.nullable<!db.decimal<32, 6>>})] (%arg1: !relalg.tuplestream,%arg2: !relalg.tuple){
        %39 = relalg.aggrfn min @partsupp1::@ps_supplycost %arg1 : !db.nullable<!db.decimal<32, 6>>
        relalg.return %39 : !db.nullable<!db.decimal<32, 6>>
      }
      %34 = relalg.getscalar @aggr0::@min %33 : !db.nullable<!db.decimal<32, 6>>
      %35 = db.cast %20 : !db.decimal<12, 2> -> !db.decimal<32, 6>
      %36 = db.as_nullable %35 : !db.decimal<32, 6> -> <!db.decimal<32, 6>>
      %37 = db.compare eq %36 : !db.nullable<!db.decimal<32, 6>>, %34 : !db.nullable<!db.decimal<32, 6>>
      %38 = db.derive_truth %37 : !db.nullable<i1>
      relalg.return %38 : i1
    }
    %13 = relalg.join %9, %12 (%arg0: !relalg.tuple){
      %true = arith.constant true
      relalg.return %true : i1
    }
    %14 = relalg.projection all [@part::@p_partkey,@part::@p_mfgr,@partsupp::@ps_suppkey] %13
    %15 = relalg.join %7, %14 (%arg0: !relalg.tuple){
      %true = arith.constant true
      relalg.return %true : i1
    }
    %16 = relalg.projection all [@supplier::@s_acctbal,@supplier::@s_name,@nation::@n_name,@part::@p_partkey,@part::@p_mfgr,@supplier::@s_address,@supplier::@s_phone,@supplier::@s_comment] %15
    %17 = relalg.sort %16 [(@supplier::@s_acctbal,desc),(@nation::@n_name,asc),(@supplier::@s_name,asc),(@part::@p_partkey,asc)]
    %18 = relalg.limit 100 %17
    %19 = relalg.materialize %18 [@supplier::@s_acctbal,@supplier::@s_name,@nation::@n_name,@part::@p_partkey,@part::@p_mfgr,@supplier::@s_address,@supplier::@s_phone,@supplier::@s_comment] => ["s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr", "s_address", "s_phone", "s_comment"] : !dsa.table
    return %19 : !dsa.table
  }
}'

echo "========================================="
echo "Testing TPC-H Query 2 with RelAlg"
echo "Expected: 4 rows"
echo "========================================="

./tools/run-relalg.sh \
  's_acctbal DECIMAL, s_name TEXT, n_name TEXT, p_partkey INTEGER, p_mfgr TEXT, s_address TEXT, s_phone TEXT, s_comment TEXT' \
  "$MLIR"
