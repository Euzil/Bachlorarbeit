hold on;
x = categorical({'clean','BadNet','Trojan Attack','TrojanNet'});
x = reordercats(x,{'clean','BadNet','Trojan Attack','TrojanNet'});
y = [1.153 3.171 2.073 1.558];
threshold = 2;
bar(x,y)
plot(threshold)
