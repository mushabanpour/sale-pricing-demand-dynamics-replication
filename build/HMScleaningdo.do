version 17
clear all
set more off

*------------------------------------------------------------
* Paths and settings
*------------------------------------------------------------
* Run from the package folder:  stata -b do build/HMScleaningdo.do
local hms    "/path/to/nielsen_extracts/HMS"       // the raw Homescan extract
local outdir "build/work"                           // holds upc_lookup_exact_container.parquet

local lookup_parquet "`outdir'/upc_lookup_exact_container.parquet"
local years 2016 2017 2018 2019
local stringcols "1 2"

tempfile upc_lookup purchases_all trips_all

*------------------------------------------------------------
* 1. Import and save UPC lookup
*------------------------------------------------------------
import parquet using "`lookup_parquet'", clear

* Make sure UPC type is consistent for merging
capture confirm string variable upc
if _rc {
    tostring upc, replace format(%20.0f)
}
replace upc = strtrim(upc)

* m:1 merge requires unique UPCs in lookup file
capture isid upc
if _rc {
    di as error "UPC lookup has duplicate upc values. m:1 merge is not valid."
    duplicates report upc
    exit 459
}

compress
save "`upc_lookup'", replace

*------------------------------------------------------------
* 2. Import purchases by year, merge with UPC lookup, append
*------------------------------------------------------------
local first 1

foreach y of local years {

    local purchase_file "`hms'/`y'/Annual_Files/purchases_`y'.tsv"

    capture confirm file "`purchase_file'"
    if _rc {
        di as error "Missing file: `purchase_file'"
        exit 601
    }

    di as txt "Processing purchases for `y'..."

    import delimited using "`purchase_file'", clear stringcols(`stringcols') varnames(1)

    capture confirm variable upc
    if _rc {
        di as error "Variable upc not found in `purchase_file'"
        exit 111
    }

    capture confirm string variable upc
    if _rc {
        tostring upc, replace format(%20.0f)
    }
    replace upc = strtrim(upc)

    merge m:1 upc using "`upc_lookup'", keep(match) nogen

    capture confirm variable panel_year
    if _rc {
        gen int panel_year = `y'
    }
    else {
        replace panel_year = `y' if missing(panel_year)
    }

    compress

    if `first' {
        save "`purchases_all'", replace
        local first 0
    }
    else {
        append using "`purchases_all'"
        save "`purchases_all'", replace
    }
}

use "`purchases_all'", clear
order panel_year upc, first
save "`outdir'/purchases_matched_2016_2019.dta", replace

*------------------------------------------------------------
* 3. Import trips by year and append
*------------------------------------------------------------
local tripvars ///
    trip_code_uc household_code purchase_date retailer_code ///
    store_code_uc panel_year store_zip3 total_spent method_of_payment_cd

local first 1

foreach y of local years {

    local trip_file "`hms'/`y'/Annual_Files/trips_`y'.tsv"

    capture confirm file "`trip_file'"
    if _rc {
        di as error "Missing file: `trip_file'"
        exit 601
    }

    di as txt "Processing trips for `y'..."

    import delimited using "`trip_file'", clear stringcols(`stringcols') varnames(1)

    capture confirm variable panel_year
    if _rc {
        gen int panel_year = `y'
    }
    else {
        replace panel_year = `y' if missing(panel_year)
    }

    foreach v of local tripvars {
        capture confirm variable `v'
        if _rc {
            di as error "Variable `v' not found in `trip_file'"
            exit 111
        }
    }

    keep `tripvars'
    compress

    if `first' {
        save "`trips_all'", replace
        local first 0
    }
    else {
        append using "`trips_all'"
        save "`trips_all'", replace
    }
}

use "`trips_all'", clear
order panel_year household_code trip_code_uc, first
save "`outdir'/trips_2016_2019.dta", replace


use "`purchases_all'", clear
merge m:1 trip_code_uc panel_year using "`trips_all'", ///
    keep(master match) nogen
save "`outdir'/purchases_with_trips_2016_2019.dta", replace
