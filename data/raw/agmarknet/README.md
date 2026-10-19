# Put downloaded price files here

Any mix of `.xlsx`, `.xls`, `.csv` or `.html` from Agmarknet or CEDA works.
Subfolders are fine.

Name each file `<commodity>_<market>_<years>`, for example:

- `potato_deesa-veg-yard_2015-2026.xlsx`
- `castor_deesa_2015-2020.xls`
- `groundnut_palanpur_2021-2026.csv`
- `ceda_castor_gujarat.csv` (put `ceda` in the name for CEDA files)

The name is used only when the table itself has no Commodity or Market column.

Then run: `python -m krishibhav load-files`
