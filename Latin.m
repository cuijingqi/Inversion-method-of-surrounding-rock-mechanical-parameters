
clc;clear

pd=cell(1,2,3);
pd{1} = makedist('normal','mu',1.25,'sigma',0.75);
pd{2} = makedist('normal','mu',26.5,'sigma',4.5);
pd{3} = makedist('normal','mu',0.2,'sigma',0.1);
correlation = [1 0 0;0 1 0;0 0 1];
n = 2500;
correlatedSamples = lhsgeneral(pd,correlation,n);



